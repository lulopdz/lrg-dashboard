"""Forecast tomorrow's DAM (Day-Ahead Market) price for one zone.

Tomorrow's DAM is genuinely unpublished at the time this normally runs (IESO's
day-ahead auction clears in the afternoon, hours after the 6am dashboard refresh),
so this is a real prediction problem, not a re-derivation of known data.

Model: gradient-boosted trees over calendar features (hour/day-of-week/month),
IESO's own load & wind forecasts (already published 2-3 days ahead), Ottawa
weather forecast, and same-hour price lags/rolling means. Also finds the single
most similar historical day (by forecasted load/wind/weather/weekend-ness) and
reports its actual price curve as an analog reference.

Run manually: `python src/predict_dam.py`. Writes data/dam_forecast.csv and
data/dam_forecast_meta.json, which generar_web.py reads if present.
"""
import json

import pandas as pd

from forecast_common import (
    ANALOG_FEATURE_LABELS, BACKTEST_DAYS, DATA_DIR, ZONE,
    add_lag_features, backtest, build_grid, determine_target_date,
    find_similar_day, fit_final_model, load_forecast_inputs, load_price_series,
    recommend_hour,
)

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "dam_lag_1d", "dam_lag_7d", "dam_roll_7d", "dam_roll_28d",
]


def main():
    dam = load_price_series("ieso_dam_prices.csv")
    load_fc, wind_fc, weather = load_forecast_inputs()
    target_date = determine_target_date(dam)
    tz = dam["interval_start_local"].dt.tz

    df = build_grid(dam, load_fc, wind_fc, weather, target_date, tz)
    df = add_lag_features(df, prefix="dam")
    df_hist = df[df["lmp"].notna()].copy()
    df_target = df[df["interval_start_local"].dt.date == target_date].copy()

    print(f"Target date (tomorrow, not yet published): {target_date}")
    print(f"Training rows: {len(df_hist)} hourly observations through {df_hist['interval_start_local'].max()}")

    metrics = backtest(df_hist, FEATURE_COLS, naive_col="dam_lag_7d")
    if metrics:
        print(f"Backtest (last {BACKTEST_DAYS}d): model MAE ${metrics['model_mae']:.2f} vs. "
              f"naive-lag-7d MAE ${metrics['naive_mae']:.2f} (RMSE ${metrics['model_rmse']:.2f})")
    else:
        print("Not enough history yet for a holdout backtest.")

    model, usable_cols = fit_final_model(df_hist, FEATURE_COLS)
    missing_features = df_target[FEATURE_COLS].isna().any(axis=1).sum()
    if missing_features:
        print(f"Warning: {missing_features} of {len(df_target)} target hours have missing inputs "
              "(forecast not published that far out yet) -- predictions for those hours are less reliable.")
    df_target["predicted_lmp"] = model.predict(df_target[usable_cols])

    best_hour, best_hour_mae = recommend_hour(metrics, df_target, FEATURE_COLS)
    if best_hour:
        print(f"Most confident hour: {best_hour} (historical backtest MAE ${best_hour_mae:.2f})")

    analog = find_similar_day(df, df_hist, target_date)
    analog_curve = {}
    analog_date_str = None
    similarity_distance = None
    comparison = {}
    if analog:
        analog_date, similarity_distance, comparison = analog
        analog_date_str = str(analog_date)
        analog_rows = df_hist[df_hist["interval_start_local"].dt.date == analog_date]
        analog_curve = dict(zip((analog_rows["hour"] + 1).tolist(), analog_rows["lmp"].tolist()))
        print(f"Most similar historical day: {analog_date} (distance {similarity_distance:.2f})")
        for col, (label, unit) in ANALOG_FEATURE_LABELS.items():
            print(f"  {label}: tomorrow's forecast {comparison[col]['target']:.1f}{unit} "
                  f"vs. {analog_date} {comparison[col]['analog']:.1f}{unit}")
    else:
        print("Not enough complete historical days to find a similar-day analog.")

    out = pd.DataFrame({
        "hour": (df_target["hour"] + 1).values,
        "predicted_lmp": df_target["predicted_lmp"].round(2).values,
    })
    out["analog_lmp"] = out["hour"].map(analog_curve).round(2)
    out_path = DATA_DIR / "dam_forecast.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")

    comparison_display = [
        {"label": label, "unit": unit, "target": comparison[col]["target"], "analog": comparison[col]["analog"]}
        for col, (label, unit) in ANALOG_FEATURE_LABELS.items()
    ] if comparison else []

    meta = {
        "zone": ZONE,
        "target_date": str(target_date),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "analog_date": analog_date_str,
        "analog_distance": similarity_distance,
        "analog_comparison": comparison_display,
        "recommended_hour": {"hour": best_hour, "expected_error": best_hour_mae} if best_hour else None,
        "backtest": metrics,
    }
    meta_path = DATA_DIR / "dam_forecast_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
