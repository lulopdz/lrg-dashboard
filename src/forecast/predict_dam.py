"""Forecast tomorrow's DAM (Day-Ahead Market) price for one zone.

Tomorrow's DAM is genuinely unpublished at the time this normally runs (IESO's
day-ahead auction clears in the afternoon, hours after the 6am dashboard refresh),
so this is a real prediction problem, not a re-derivation of known data.

Model: gradient-boosted trees over calendar features (hour/day-of-week/month),
IESO's own load & wind forecasts (already published 2-3 days ahead), Ottawa
weather forecast, and same-hour price lags/rolling means. Also finds the single
most similar historical day (by forecasted load/wind/weather/weekend-ness) and
reports its actual price curve as an analog reference.

Run manually: `python src/forecast/predict_dam.py`. Writes data/dam_forecast.csv and
data/dam_forecast_meta.json, which generar_web.py reads if present.
"""
from forecast_common import load_price_series, run_forecast

FEATURE_COLS = [
    # raw `hour` sits alongside hour_sin/cos so a tree can split directly on hour of day
    # instead of reconstructing it from the two smooth features
    "hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "wind_speed_100m_port_alma",  # wind at turbine hub height, Chatham-Kent (see update_weather.py)
    "dam_lag_1d", "dam_lag_7d", "dam_roll_7d", "dam_roll_28d",
]

# DAM opts out of forecast_common.DEFAULT_MODEL_PARAMS' depth cap: it's the cleanest of the
# three series and the extra regularization bought nothing here on a 6-fold walk-forward
# ($11.46 -> $11.48, winning only 4 of 6 folds), while the feature changes alone took it
# $11.46 -> $10.64. {} means scikit-learn's own defaults.
MODEL_PARAMS = {}


def main():
    dam = load_price_series("ieso_dam_prices.csv")
    run_forecast("dam", dam, FEATURE_COLS, dam, model_params=MODEL_PARAMS)


if __name__ == "__main__":
    main()
