"""Forecast tomorrow's RTM (Real-Time Market) price for one zone.

Unlike DAM, RTM stays genuinely unknown even after the day-ahead auction clears --
it's the actual real-time settlement price, driven by real-time system conditions
DAM couldn't fully anticipate. This is what the dashboard's Spread tab (DAM - RTM)
ultimately wants forecasted: knowing tomorrow's expected spread ahead of time.

Model: same calendar + IESO load/wind forecast + Ottawa weather forecast features as
predict_dam.py, plus the DAM price for the same hour as an extra feature -- DAM already
prices in the market's day-ahead expectation of supply/demand balance, so it's normally
the single strongest predictor of RTM. For historical hours this is the *actual*
published DAM; for tomorrow (not yet published) it falls back to predict_dam.py's own
prediction if data/dam_forecast.csv exists. Run predict_dam.py first for best results --
this script still runs without it, just with a weaker feature for tomorrow's hours.

Run manually: `python src/predict_rtm.py`. Writes data/rtm_forecast.csv and
data/rtm_forecast_meta.json, which generar_web.py reads if present.
"""
from forecast_common import load_price_series, run_forecast

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "dam_price",
    "rtm_lag_1d", "rtm_lag_7d", "rtm_roll_7d", "rtm_roll_28d",
]


def main():
    rtm = load_price_series("ieso_rtm_prices.csv")
    dam = load_price_series("ieso_dam_prices.csv")
    run_forecast("rtm", rtm, FEATURE_COLS, dam, attach_dam_feature=True)


if __name__ == "__main__":
    main()
