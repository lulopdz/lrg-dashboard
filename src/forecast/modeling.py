"""Model training, holdout backtesting, quantile banding, and the run_forecast driver."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from analog import analog_comparison_display, analog_price_curves, find_similar_day
from config import (
    ANALOG_FEATURE_LABELS,
    BACKTEST_DAYS,
    BACKTEST_SERIES_KEYS,
    DATA_DIR,
    DEFAULT_MODEL_PARAMS,
    MIN_TRAIN_DAYS,
    QUANTILES,
    ZONE,
    forecast_file,
    zone_feature_cols,
)
from features import (
    add_lag_features,
    attach_reference_price,
    build_grid,
    missing_inputs,
    usable_feature_cols,
)
from io_archive import (
    archive_forecast,
    dam_target_prediction,
    determine_target_date,
    information_cutoff,
    load_forecast_inputs,
    load_supply_inputs,
)


def training_target(df, target_clip=None):
    """The series a point model is fit on: 'lmp', capped at +/-target_clip when given. Only
    the fit sees the cap; every error is still scored against the real price."""
    return df["lmp"] if target_clip is None else df["lmp"].clip(-target_clip, target_clip)


def backtest(df_hist, feature_cols, naive_col, model_params=None, target_clip=None):
    """Trailing holdout: honest accuracy vs. a naive 'same hour last week' baseline, plus
    a per-hour-of-day error breakdown (which hours the model has historically nailed vs.
    missed) used to recommend the hour we're most confident in. model_params and target_clip
    must match what fit_final_model uses, or the reported accuracy won't describe the shipped
    model."""
    cutoff = df_hist["interval_start_local"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    train = df_hist[df_hist["interval_start_local"] < cutoff]
    test = df_hist[df_hist["interval_start_local"] >= cutoff]
    if len(train) < MIN_TRAIN_DAYS * 24 or test.empty:
        return None

    usable_cols = usable_feature_cols(train, feature_cols)
    model = HistGradientBoostingRegressor(random_state=0, **(model_params or {}))
    model.fit(train[usable_cols], training_target(train, target_clip))
    pred = model.predict(test[usable_cols])

    naive = test[naive_col].fillna(train["lmp"].mean())

    abs_err = np.abs(test["lmp"].values - pred)
    hourly_mae = (
        pd.DataFrame({"hour": test["hour"].values + 1, "abs_err": abs_err})
        .groupby("hour")["abs_err"].mean()
        .reindex(range(1, 25))
    )

    return {
        "n_test_hours": int(len(test)),
        "model_mae": float(mean_absolute_error(test["lmp"], pred)),
        "model_rmse": float(np.sqrt(mean_squared_error(test["lmp"], pred))),
        "naive_mae": float(mean_absolute_error(test["lmp"], naive)),
        "hourly_mae": {int(h): (None if pd.isna(v) else float(v)) for h, v in hourly_mae.items()},
        # Raw holdout series, kept so a caller (see directional_backtest) can re-score the same
        # window as Long/Short calls rather than an error magnitude. Stripped by run_forecast
        # before the meta dict is written -- they're ~500 values each and only the derived
        # summary is ever read back, so persisting them just bloats a file the bot commits daily.
        BACKTEST_SERIES_KEYS[0]: [float(v) for v in test["lmp"].values],
        BACKTEST_SERIES_KEYS[1]: [float(v) for v in pred],
    }


def directional_backtest(metrics):
    """Scores the model's own backtest predictions exactly like the Trading Simulator scores a
    human's Long/Short calls (see generar_simulator.py's reveal()): a predicted value > 0 is a
    Long call, < 0 is Short; PnL is +actual on a correct call, -actual on a wrong one (actual
    can itself be negative); optimal PnL is sum(abs(actual)) -- what a perfect-hindsight caller
    would have made. Reframes 'how accurate is this model' as a track record ('you'd have gone
    14-7 and made $342 of $510 possible') instead of a dollar error, which answers 'should I
    trust tomorrow's call' more directly than an MAE number does. Sign-based, no deadband --
    the model always has an opinion, same as it always outputs some nonzero predicted value.
    Only called for the spread predictor; DAM/RTM prices don't have a natural direction to
    call. Returns None if there's no backtest to score."""
    if not metrics or not metrics.get("test_actual"):
        return None
    actual = np.asarray(metrics["test_actual"], dtype=float)
    predicted = np.asarray(metrics["test_predicted"], dtype=float)
    pnl = np.where(predicted > 0, actual, -actual)
    optimal_pnl = float(np.abs(actual).sum())
    total_pnl = float(pnl.sum())
    correct = int((pnl > 0).sum())
    n = len(actual)

    def pct(value):
        return round(value / optimal_pnl * 100, 1) if optimal_pnl else None

    # The bar the model has to clear isn't zero -- it's "pick one side and never change your
    # mind". The spread is positive ~66% of hours, so always-Virtual-Gen wins most hours
    # outright; whether it also makes money depends on the period, since the negative hours
    # are the big ones. Reporting both keeps a mediocre-looking win rate honest in each
    # direction: a model can win fewer hours than always-Gen and still make far more money,
    # or beat it on win rate while losing to always-Load on P&L.
    always_gen = float(actual.sum())
    return {
        "n_hours": n,
        "correct": correct,
        "win_rate": round(correct / n * 100, 1) if n else None,
        "total_pnl": round(total_pnl, 2),
        "optimal_pnl": round(optimal_pnl, 2),
        "pct_of_optimal": pct(total_pnl),
        "naive_gen_pnl": round(always_gen, 2),
        "naive_gen_pct": pct(always_gen),
        "naive_gen_win_rate": round(float((actual > 0).mean()) * 100, 1) if n else None,
        "naive_load_pnl": round(-always_gen, 2),
        "naive_load_pct": pct(-always_gen),
        "naive_load_win_rate": round(float((actual < 0).mean()) * 100, 1) if n else None,
    }


def recommend_hour(metrics, df_target, feature_cols):
    """The hour (1-24) with the lowest historical backtest error, restricted to hours
    whose target-day inputs are actually complete -- what we surface as 'most confident
    hour'. Returns (hour, expected_mae) or (None, None) if there's nothing to go on."""
    if not metrics or not metrics.get("hourly_mae"):
        return None, None

    missing_mask = df_target[feature_cols].isna().any(axis=1)
    missing_hours = set((df_target.loc[missing_mask, "hour"] + 1).tolist())

    candidates = {h: mae for h, mae in metrics["hourly_mae"].items() if mae is not None and h not in missing_hours}
    if not candidates:
        candidates = {h: mae for h, mae in metrics["hourly_mae"].items() if mae is not None}
    if not candidates:
        return None, None

    best_hour = min(candidates, key=candidates.get)
    return best_hour, candidates[best_hour]


def order_band(df_target):
    """P10 <= point <= P90. The three boosters are fit independently, so nothing stops a
    quantile from landing on the wrong side of the point forecast (or of each other) in an
    hour with little training support; a band that doesn't contain its own centre reads as a
    bug on the chart."""
    lo = df_target[["p10", "p90", "predicted_lmp"]].min(axis=1)
    hi = df_target[["p10", "p90", "predicted_lmp"]].max(axis=1)
    return df_target.assign(p10=lo, p90=hi)


def fit_quantile_model(df_hist, usable_cols, quantile, model_params=None):
    params = {k: v for k, v in (model_params or {}).items() if k != "loss"}
    model = HistGradientBoostingRegressor(random_state=0, loss="quantile", quantile=quantile, **params)
    model.fit(df_hist[usable_cols], df_hist["lmp"])
    return model


def fit_final_model(df_hist, feature_cols, model_params=None, target_clip=None):
    """Returns (model, usable_cols) -- usable_cols is feature_cols minus anything entirely
    missing in df_hist; predict() calls must select the same columns, not the original list."""
    usable_cols = usable_feature_cols(df_hist, feature_cols)
    model = HistGradientBoostingRegressor(random_state=0, **(model_params or {}))
    model.fit(df_hist[usable_cols], training_target(df_hist, target_clip))
    return model, usable_cols


def run_forecast(prefix, series_df, feature_cols, dam, attach_dam_feature=False,
                 model_params=None, vintage="pre_dam", target_date=None, out_dir=DATA_DIR,
                 dam_forecast_dir=DATA_DIR, backtest_enabled=True, zone=ZONE, target_clip=None):
    """End-to-end driver shared by predict_dam.py/predict_rtm.py/predict_spread.py: builds the
    hourly grid, optionally attaches the same-hour DAM price as a feature, backtests, fits the
    final model, predicts the target day, finds a similar-day analog, and writes
    {out_dir}/{prefix}_forecast.csv + {prefix}_forecast_meta.json + the history archive.

    prefix names the lag features (e.g. 'dam' -> dam_lag_2d, used as the naive baseline too)
    and the output files. series_df is the series being predicted (DAM/RTM price, or the
    DAM-RTM spread); dam is always the DAM price series, used both to anchor the target day
    and, when attach_dam_feature=True, merged in as the 'dam_price' feature (predict_rtm.py/
    predict_spread.py's FEATURE_COLS must include it; predict_dam.py doesn't need it).
    model_params overrides DEFAULT_MODEL_PARAMS for this series (pass {} for scikit-learn's
    own defaults); the same value is used for both the backtest and the shipped model so the
    reported accuracy describes what actually produced the forecast.

    vintage is pre_dam or post_dam (see VINTAGES). target_date=None means the live target for
    that vintage; a past date is a walk-forward reconstruction: history is cut at that run's
    information cutoff and the DAM feature comes from the archived vintage in dam_forecast_dir,
    so the result is what the live run would have produced (walkforward.py drives this)."""
    model_params = DEFAULT_MODEL_PARAMS if model_params is None else model_params
    feature_cols = zone_feature_cols(feature_cols, zone)
    load_fc, wind_fc, weather = load_forecast_inputs()
    tz = series_df["interval_start_local"].dt.tz
    reconstruction = target_date is not None
    if reconstruction:
        if prefix == "dam":  # published a full day at once: all of today is known at the run
            series_df = series_df[series_df["interval_start_local"].dt.date < target_date]
        else:  # RT-based: only today's first RT_HOURS_KNOWN hours are settled
            series_df = series_df[series_df["interval_start_local"] < information_cutoff(target_date, vintage, tz)]
        if prefix == "dam" or vintage == "pre_dam":  # DAM is known through today only
            dam = dam[dam["interval_start_local"].dt.date < target_date]
        else:  # post_dam: target day's DAM is known, nothing beyond it
            dam = dam[dam["interval_start_local"].dt.date <= target_date]
    else:
        target_date = determine_target_date(dam, vintage)

    if not reconstruction and vintage == "pre_dam" and prefix == "dam" and zone == ZONE:
        # Keep today's view of tomorrow's inputs before anything overwrites it (vintages.py).
        from vintages import archive as archive_dayahead, inputs_for_day
        archive_dayahead(inputs_for_day(target_date, load_fc, wind_fc, weather, load_supply_inputs()))

    df = build_grid(series_df, load_fc, wind_fc, weather, target_date, tz)
    used_dam_forecast = None
    if attach_dam_feature:
        target_pred = dam_target_prediction(target_date, vintage, dam_forecast_dir, zone)
        df, used_dam_forecast = attach_reference_price(df, dam, target_date, target_pred)
    df = add_lag_features(df, prefix=prefix, known_through_yesterday=(prefix == "dam"), vintage=vintage)
    df_hist = df[df["lmp"].notna()].copy()
    df_target = df[df["interval_start_local"].dt.date == target_date].copy()

    print(f"Target date: {target_date} ({vintage}{', reconstruction' if reconstruction else ''})")
    print(f"Training rows: {len(df_hist)} hourly observations through {df_hist['interval_start_local'].max()}")
    if attach_dam_feature and vintage == "pre_dam" and not target_pred:
        print("Note: no DAM forecast for the target day -- run predict_dam.py first for a same-hour DAM "
              "feature; the target hours will have no DAM info this run.")

    metrics = (backtest(df_hist, feature_cols, naive_col=f"{prefix}_lag_7d", model_params=model_params,
                        target_clip=target_clip)
               if backtest_enabled else None)
    if metrics:
        print(f"Backtest (last {BACKTEST_DAYS}d): model MAE ${metrics['model_mae']:.2f} vs. "
              f"naive-lag-7d MAE ${metrics['naive_mae']:.2f} (RMSE ${metrics['model_rmse']:.2f})")
    else:
        print("Not enough history yet for a holdout backtest.")

    track_record = directional_backtest(metrics) if prefix == "spread" else None
    if metrics:  # scored above; drop the raw series so they don't land in the meta JSON
        for key in BACKTEST_SERIES_KEYS:
            metrics.pop(key, None)
    if track_record:
        print(f"Directional track record: {track_record['correct']}-{track_record['n_hours'] - track_record['correct']} "
              f"({track_record['win_rate']:.0f}% win rate), ${track_record['total_pnl']:.0f} of "
              f"${track_record['optimal_pnl']:.0f} possible ({track_record['pct_of_optimal']:.0f}% of optimal)")

    model, usable_cols = fit_final_model(df_hist, feature_cols, model_params=model_params, target_clip=target_clip)
    missing_features, missing_cols = missing_inputs(df_target, feature_cols)
    df_target["predicted_lmp"] = model.predict(df_target[usable_cols])
    for q in QUANTILES:
        df_target[f"p{int(q * 100)}"] = fit_quantile_model(df_hist, usable_cols, q, model_params).predict(df_target[usable_cols])
    df_target = order_band(df_target)

    best_hour, best_hour_mae = recommend_hour(metrics, df_target, feature_cols)
    if best_hour:
        print(f"Most confident hour: {best_hour} (historical backtest MAE ${best_hour_mae:.2f})")

    analogs = find_similar_day(df, df_hist, target_date)
    analog_curves = analog_price_curves(analogs, df_hist)

    if analogs:
        for rank, a in enumerate(analogs, start=1):
            print(f"#{rank} similar historical day: {a['date']} (distance {a['distance']:.2f})")
        for col, (label, unit) in ANALOG_FEATURE_LABELS.items():
            target_val = analogs[0]["comparison"][col]["target"]
            analog_vals = " / ".join(f"{a['comparison'][col]['analog']:.1f}{unit}" for a in analogs)
            print(f"  {label}: tomorrow's forecast {target_val:.1f}{unit} vs. {analog_vals}")
    else:
        print("Not enough complete historical days to find a similar-day analog.")

    out = pd.DataFrame({
        "hour": (df_target["hour"] + 1).values,
        "predicted_lmp": df_target["predicted_lmp"].round(2).values,
    })
    for q in QUANTILES:
        col = f"p{int(q * 100)}"
        out[col] = df_target[col].round(2).values
    out["analog_lmp"] = out["hour"].map(analog_curves[0] if analogs else {}).round(2)
    out["analog_lmp_2"] = out["hour"].map(analog_curves[1] if len(analogs) > 1 else {}).round(2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / forecast_file(prefix, "forecast", zone)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")

    meta = {
        "zone": zone,
        "target_date": str(target_date),
        "vintage": vintage,
        "reconstruction": reconstruction,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "missing_input_hours": int(missing_features),
        "missing_input_cols": missing_cols,
        "analog_date": str(analogs[0]["date"]) if analogs else None,
        "analog_distance": analogs[0]["distance"] if analogs else None,
        "analog_comparison": analog_comparison_display(analogs, 0),
        "analog_date_2": str(analogs[1]["date"]) if len(analogs) > 1 else None,
        "analog_distance_2": analogs[1]["distance"] if len(analogs) > 1 else None,
        "analog_comparison_2": analog_comparison_display(analogs, 1),
        "recommended_hour": {"hour": best_hour, "expected_error": best_hour_mae} if best_hour else None,
        "backtest": metrics,
        "directional_backtest": track_record,
    }
    if attach_dam_feature:
        meta["used_dam_forecast_feature"] = used_dam_forecast
    meta_path = out_dir / forecast_file(prefix, "forecast_meta", zone)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")
    archive_forecast(prefix, out, meta, out_dir=out_dir, zone=zone)
    return out, meta
