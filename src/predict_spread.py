"""Forecast tomorrow's DAM-RTM spread for one zone -- the actual tradeable signal this
dashboard's Spread tab is built around: does the day-ahead price over- or under-shoot
what real-time conditions end up delivering, and by how much?

Modeled directly on the historical spread series (not just predicted_DAM - predicted_RTM
from the other two scripts), since the spread has its own shape -- which hours tend to
diverge, and how much -- distinct from either price series alone, and fitting it directly
avoids compounding two separate models' errors in an uncontrolled way.

Same calendar + IESO load/wind forecast + Ottawa weather forecast features as
predict_dam.py/predict_rtm.py, plus the same-hour DAM price (actual historically,
predict_dam.py's own prediction for tomorrow) since spread magnitude often tracks the
price level itself. Run predict_dam.py first for the best results on tomorrow's hours.

Run manually: `python src/predict_spread.py`. Writes data/spread_forecast.csv and
data/spread_forecast_meta.json, which generar_web.py reads if present.
"""
from forecast_common import load_price_series, run_forecast

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "dam_price",
    "spread_lag_1d", "spread_lag_7d", "spread_roll_7d", "spread_roll_28d",
]


def compute_spread_series(dam, rtm):
    merged = dam.merge(rtm, on="interval_start_local", suffixes=("_dam", "_rtm"))
    merged["lmp"] = merged["lmp_dam"] - merged["lmp_rtm"]
    return merged[["interval_start_local", "lmp"]]


def main():
    dam = load_price_series("ieso_dam_prices.csv")
    rtm = load_price_series("ieso_rtm_prices.csv")
    spread = compute_spread_series(dam, rtm)
    run_forecast("spread", spread, FEATURE_COLS, dam, attach_dam_feature=True)


if __name__ == "__main__":
    main()
