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
from forecast_common import load_price_series, run_forecast

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "dam_lag_1d", "dam_lag_7d", "dam_roll_7d", "dam_roll_28d",
]


def main():
    dam = load_price_series("ieso_dam_prices.csv")
    run_forecast("dam", dam, FEATURE_COLS, dam)


if __name__ == "__main__":
    main()
