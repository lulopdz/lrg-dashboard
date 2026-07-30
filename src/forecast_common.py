"""Shared plumbing for the next-day DAM/RTM/Spread predictors (predict_dam.py, predict_rtm.py,
predict_spread.py): loading IESO's own forecasts, calendar feature engineering, backtesting,
the similar-day analog search, and the end-to-end run_forecast() driver they all call."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ZONE = "OTTAWA"
WIND_ZONE = "Ontario Total"
BACKTEST_DAYS = 21   # trailing holdout window used to report honest accuracy
MIN_TRAIN_DAYS = 60  # need enough history before the holdout window to bother training

# The full set of forecasted variables we have for tomorrow (IESO load forecast +
# IESO wind forecast + Open-Meteo weather forecast) -- "similar day" means closest
# on these, not on anything realized/actual. Each entry is
# (feature_key, source_column, agg_func, label, unit, weight): source_column/agg_func
# say how to build the daily value from the hourly grid (temperature_2m_max reuses
# temperature_2m's column with "max" instead of "mean"); weight scales that feature's
# contribution to the distance in find_similar_day -- load and wind forecasts move
# price the most, so they count double; temperature (avg and max) drives demand too
# but a bit less directly, so it's 1.5x; the rest of the weather variables stay at 1x.
_ANALOG_SPEC = [
    ("ontario", "ontario", "mean", "Ontario load forecast", "MW", 2.0),
    ("ontario_southeast", "ontario_southeast", "mean", "SE Ontario load forecast", "MW", 2.0),
    ("wind_forecast", "wind_forecast", "mean", "Wind generation forecast", "MW", 2.0),
    ("temperature_2m", "temperature_2m", "mean", "Temperature", "°C", 1.5),
    ("temperature_2m_max", "temperature_2m", "max", "Max temperature", "°C", 1.5),
    ("wind_speed_10m", "wind_speed_10m", "mean", "Wind speed", "m/s", 1.0),
    ("precipitation", "precipitation", "mean", "Precipitation", "mm", 1.0),
    ("snowfall", "snowfall", "mean", "Snowfall", "cm", 1.0),
    ("relative_humidity_2m", "relative_humidity_2m", "mean", "Humidity", "%", 1.0),
    ("shortwave_radiation", "shortwave_radiation", "mean", "Solar radiation", "W/m²", 1.0),
]

ANALOG_FEATURE_COLS = [key for key, *_ in _ANALOG_SPEC]
ANALOG_FEATURE_LABELS = {key: (label, unit) for key, _src, _agg, label, unit, _w in _ANALOG_SPEC}
ANALOG_FEATURE_WEIGHTS = {key: w for key, _src, _agg, _label, _unit, w in _ANALOG_SPEC}


def load_price_series(filename, zone=ZONE):
    """DAM or RTM hourly price for one zone -- both CSVs share the same
    interval_start_local / location / lmp schema."""
    df = pd.read_csv(DATA_DIR / filename, parse_dates=["interval_start_local"])
    df = df[df["location"] == zone][["interval_start_local", "lmp"]].sort_values("interval_start_local")
    return df


def load_forecast_inputs():
    """The exogenous forecasts available for tomorrow, common to both predictors."""
    load_cols = ["ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast"]
    load_fc = pd.read_csv(DATA_DIR / "ieso_load_forecast.csv", parse_dates=["interval_start_local"])
    load_fc = load_fc[["interval_start_local"] + load_cols].drop_duplicates("interval_start_local", keep="last")

    wind_fc = pd.read_csv(DATA_DIR / "ieso_wind_forecast.csv", parse_dates=["interval_start_local"])
    wind_fc = wind_fc[wind_fc["zone"] == WIND_ZONE][["interval_start_local", "generation_forecast"]]
    wind_fc = wind_fc.rename(columns={"generation_forecast": "wind_forecast"}).drop_duplicates("interval_start_local", keep="last")

    weather = pd.read_csv(DATA_DIR / "OTTAWA_weather.csv", parse_dates=["timestamp"])
    weather = weather.rename(columns={"timestamp": "interval_start_local"}).drop_duplicates("interval_start_local", keep="last")

    return load_fc, wind_fc, weather


def determine_target_date(dam):
    """Tomorrow, anchored on DAM (published as a full day at once) so the DAM and RTM
    predictors always target the same day -- otherwise a predicted spread wouldn't line up."""
    return (dam["interval_start_local"].max() + pd.Timedelta(days=1)).normalize().date()


def build_grid(target_df, load_fc, wind_fc, weather, target_date, tz):
    """One continuous hourly grid spanning history through target_date, with the target
    series ('lmp'), forecast inputs, and calendar features merged in. 'lmp' is NaN for
    target_date -- that's the row(s) to predict."""
    start = target_df["interval_start_local"].min()
    end = pd.Timestamp(target_date, tz=tz) + pd.Timedelta(hours=23)
    idx = pd.date_range(start, end, freq="h", tz=tz)

    df = pd.DataFrame({"interval_start_local": idx})
    df = df.merge(target_df, on="interval_start_local", how="left")
    df = df.merge(load_fc, on="interval_start_local", how="left")
    df = df.merge(wind_fc, on="interval_start_local", how="left")
    df = df.merge(weather, on="interval_start_local", how="left")

    df["hour"] = df["interval_start_local"].dt.hour
    df["dow"] = df["interval_start_local"].dt.dayofweek
    df["month"] = df["interval_start_local"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    for col, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    return df


def add_lag_features(df, prefix):
    """Causal same-hour lags/rolling means on df['lmp'], shifted by whole days on the
    complete hourly grid so nothing at hour t ever sees data from t or later."""
    lmp = df["lmp"]
    df[f"{prefix}_lag_1d"] = lmp.shift(24)
    df[f"{prefix}_lag_7d"] = lmp.shift(24 * 7)
    df[f"{prefix}_roll_7d"] = lmp.shift(24).rolling(24 * 7, min_periods=24 * 3).mean()
    df[f"{prefix}_roll_28d"] = lmp.shift(24).rolling(24 * 28, min_periods=24 * 7).mean()
    return df


def usable_feature_cols(df, feature_cols):
    """Drop any feature that's entirely missing in this slice. HistGradientBoostingRegressor
    handles partial missingness fine, but a column with zero non-null values can crash its
    histogram binning step on some scikit-learn versions -- e.g. a forecast source (like the
    Ontario-wide wind forecast) that only has a couple weeks of history won't have a single
    real value in an older training window, even though it's fully populated more recently."""
    return [c for c in feature_cols if df[c].notna().any()]


def backtest(df_hist, feature_cols, naive_col):
    """Trailing holdout: honest accuracy vs. a naive 'same hour last week' baseline, plus
    a per-hour-of-day error breakdown (which hours the model has historically nailed vs.
    missed) used to recommend the hour we're most confident in."""
    cutoff = df_hist["interval_start_local"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    train = df_hist[df_hist["interval_start_local"] < cutoff]
    test = df_hist[df_hist["interval_start_local"] >= cutoff]
    if len(train) < MIN_TRAIN_DAYS * 24 or test.empty:
        return None

    usable_cols = usable_feature_cols(train, feature_cols)
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(train[usable_cols], train["lmp"])
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


def fit_final_model(df_hist, feature_cols):
    """Returns (model, usable_cols) -- usable_cols is feature_cols minus anything entirely
    missing in df_hist; predict() calls must select the same columns, not the original list."""
    usable_cols = usable_feature_cols(df_hist, feature_cols)
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(df_hist[usable_cols], df_hist["lmp"])
    return model, usable_cols


def attach_reference_price(df, reference_df, target_date, forecast_csv_name, feature_name="dam_price"):
    """Same-hour reference price as a feature (e.g. DAM price, for the RTM/Spread models):
    the actual published price for history, and the matching predict_*.py script's own
    prediction for target_date, since the actual price doesn't exist there yet.
    Returns (df, used_forecast) -- used_forecast is False if that script hasn't been run."""
    df = df.merge(reference_df.rename(columns={"lmp": feature_name}), on="interval_start_local", how="left")

    forecast_path = DATA_DIR / forecast_csv_name
    used_forecast = forecast_path.exists()
    if used_forecast:
        forecast_df = pd.read_csv(forecast_path)
        hour_to_pred = dict(zip(forecast_df["hour"], forecast_df["predicted_lmp"]))
        target_mask = df["interval_start_local"].dt.date == target_date
        df.loc[target_mask, feature_name] = (df.loc[target_mask, "hour"] + 1).map(hour_to_pred)
    return df, used_forecast


def find_similar_day(df, df_hist, target_date, n=2):
    """The n historical days closest to target_date (closest first) on the full set of
    forecasted variables we have for tomorrow (load forecast, wind forecast, weather
    forecast) plus weekend-ness, weighted by ANALOG_FEATURE_WEIGHTS so load/wind/
    temperature count more than the rest. Returns a list of up to n dicts
    {date, distance, comparison} (empty if too little data to compare). `comparison`
    holds each feature's target vs. that day's value, for display."""
    agg = {key: (src, func) for key, src, func, _label, _unit, _w in _ANALOG_SPEC}
    daily = df.groupby(df["interval_start_local"].dt.date).agg(
        is_weekend=("is_weekend", "max"),
        n_hours=("interval_start_local", "count"),
        **agg,
    )
    if target_date not in daily.index or daily.loc[target_date, ANALOG_FEATURE_COLS].isna().any():
        return []

    complete_dates = set(df_hist["interval_start_local"].dt.date)
    daily_hist = daily[daily.index.isin(complete_dates) & (daily["n_hours"] == 24)].dropna(subset=ANALOG_FEATURE_COLS)
    daily_hist = daily_hist.drop(index=target_date, errors="ignore")
    if daily_hist.empty:
        return []

    mu = daily_hist[ANALOG_FEATURE_COLS].mean()
    sigma = daily_hist[ANALOG_FEATURE_COLS].std().replace(0, 1)
    z_hist = (daily_hist[ANALOG_FEATURE_COLS] - mu) / sigma
    z_target = (daily.loc[target_date, ANALOG_FEATURE_COLS] - mu) / sigma
    weights = pd.Series(ANALOG_FEATURE_WEIGHTS)[ANALOG_FEATURE_COLS]

    # Weekday/weekend demand shapes differ enough that a weekend analog for a
    # weekday target (or vice versa) should lose even if load/wind/weather happen to match.
    weekend_penalty = (daily_hist["is_weekend"] != daily.loc[target_date, "is_weekend"]).astype(float) * 3.0
    dist = np.sqrt((weights * (z_hist - z_target) ** 2).sum(axis=1)) + weekend_penalty

    analogs = []
    for date, distance in dist.nsmallest(n).items():
        comparison = {
            col: {"target": float(daily.loc[target_date, col]), "analog": float(daily_hist.loc[date, col])}
            for col in ANALOG_FEATURE_COLS
        }
        analogs.append({"date": date, "distance": float(distance), "comparison": comparison})
    return analogs


def run_forecast(prefix, series_df, feature_cols, dam, attach_dam_feature=False):
    """End-to-end driver shared by predict_dam.py/predict_rtm.py/predict_spread.py: builds the
    hourly grid, optionally attaches the same-hour DAM price as a feature, backtests, fits the
    final model, predicts tomorrow, finds a similar-day analog, and writes
    data/{prefix}_forecast.csv + data/{prefix}_forecast_meta.json.

    prefix names the lag features (e.g. 'dam' -> dam_lag_1d, used as the naive baseline too)
    and the output files. series_df is the series being predicted (DAM/RTM price, or the
    DAM-RTM spread); dam is always the DAM price series, used both to anchor "tomorrow" and,
    when attach_dam_feature=True, merged in as the 'dam_price' feature (predict_rtm.py/
    predict_spread.py's FEATURE_COLS must include it; predict_dam.py doesn't need it)."""
    load_fc, wind_fc, weather = load_forecast_inputs()
    target_date = determine_target_date(dam)
    tz = series_df["interval_start_local"].dt.tz

    df = build_grid(series_df, load_fc, wind_fc, weather, target_date, tz)
    used_dam_forecast = None
    if attach_dam_feature:
        df, used_dam_forecast = attach_reference_price(df, dam, target_date, "dam_forecast.csv")
    df = add_lag_features(df, prefix=prefix)
    df_hist = df[df["lmp"].notna()].copy()
    df_target = df[df["interval_start_local"].dt.date == target_date].copy()

    print(f"Target date (tomorrow): {target_date}")
    print(f"Training rows: {len(df_hist)} hourly observations through {df_hist['interval_start_local'].max()}")
    if attach_dam_feature and not used_dam_forecast:
        print("Note: data/dam_forecast.csv not found -- run predict_dam.py first for a same-hour DAM "
              "feature on the target day; tomorrow's hours will have no DAM info this run.")

    metrics = backtest(df_hist, feature_cols, naive_col=f"{prefix}_lag_7d")
    if metrics:
        print(f"Backtest (last {BACKTEST_DAYS}d): model MAE ${metrics['model_mae']:.2f} vs. "
              f"naive-lag-7d MAE ${metrics['naive_mae']:.2f} (RMSE ${metrics['model_rmse']:.2f})")
    else:
        print("Not enough history yet for a holdout backtest.")

    model, usable_cols = fit_final_model(df_hist, feature_cols)
    missing_features = df_target[feature_cols].isna().any(axis=1).sum()
    if missing_features:
        print(f"Warning: {missing_features} of {len(df_target)} target hours have missing inputs -- "
              "predictions for those hours are less reliable.")
    df_target["predicted_lmp"] = model.predict(df_target[usable_cols])

    best_hour, best_hour_mae = recommend_hour(metrics, df_target, feature_cols)
    if best_hour:
        print(f"Most confident hour: {best_hour} (historical backtest MAE ${best_hour_mae:.2f})")

    analogs = find_similar_day(df, df_hist, target_date)
    analog_curves = []
    for a in analogs:
        rows = df_hist[df_hist["interval_start_local"].dt.date == a["date"]]
        analog_curves.append(dict(zip((rows["hour"] + 1).tolist(), rows["lmp"].tolist())))

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
    out["analog_lmp"] = out["hour"].map(analog_curves[0] if analogs else {}).round(2)
    out["analog_lmp_2"] = out["hour"].map(analog_curves[1] if len(analogs) > 1 else {}).round(2)
    out_path = DATA_DIR / f"{prefix}_forecast.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")

    def comparison_display(rank):
        if len(analogs) <= rank:
            return []
        comparison = analogs[rank]["comparison"]
        return [
            {
                "label": label, "unit": unit, "weight": ANALOG_FEATURE_WEIGHTS[col],
                "target": comparison[col]["target"], "analog": comparison[col]["analog"],
            }
            for col, (label, unit) in ANALOG_FEATURE_LABELS.items()
        ]

    meta = {
        "zone": ZONE,
        "target_date": str(target_date),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "analog_date": str(analogs[0]["date"]) if analogs else None,
        "analog_distance": analogs[0]["distance"] if analogs else None,
        "analog_comparison": comparison_display(0),
        "analog_date_2": str(analogs[1]["date"]) if len(analogs) > 1 else None,
        "analog_distance_2": analogs[1]["distance"] if len(analogs) > 1 else None,
        "analog_comparison_2": comparison_display(1),
        "recommended_hour": {"hour": best_hour, "expected_error": best_hour_mae} if best_hour else None,
        "backtest": metrics,
    }
    if attach_dam_feature:
        meta["used_dam_forecast_feature"] = used_dam_forecast
    meta_path = DATA_DIR / f"{prefix}_forecast_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")
