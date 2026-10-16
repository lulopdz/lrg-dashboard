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

Run manually: `python src/forecast/predict_rtm.py`. Writes data/rtm_forecast.csv and
data/rtm_forecast_meta.json, which generar_web.py reads if present.
"""
from forecast_common import (
    DATA_DIR, DEFAULT_MODEL_PARAMS, SUPPLY_COLS, ZONE, load_price_series, parse_run_args, run_forecast,
)

FEATURE_COLS = [
    # raw `hour` sits alongside hour_sin/cos so a tree can split directly on hour of day
    # instead of reconstructing it from the two smooth features
    "hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "is_holiday",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "wind_speed_100m_port_alma",  # wind at turbine hub height, Chatham-Kent (see update_weather.py)
    "dam_price",
    # lags start two days back plus a summary of today's hours that are settled by run time;
    # see forecast_common.add_lag_features for why not rtm_lag_1d
    "rtm_lag_2d", "rtm_lag_7d", "rtm_roll_7d", "rtm_roll_28d", "rtm_today_early_mean", "rtm_today_early_max",
    *SUPPLY_COLS,  # nuclear/gas/hydro availability, net load, capacity margin (forecast_common)
]

# RT is fit as RT - dam_price (forecast_common.OffsetRegressor) on the absolute-error loss, i.e.
# the median of how far RT strays from the DAM in hours like this one. With squared error the
# booster chased the spike hours (5% of hours, 31% of the error) and lost to plain "RT = DAM":
# walk-forward, 92 days (2026-07-10..10-17, OTTAWA), MAE $20.40 vs $19.31 for RT = real DAM
# (post_dam) and $20.86 vs $21.16 for RT = DAM forecast (pre_dam). This: $17.91 and $19.08.
# The median runs ~$8 below the mean RT (spikes are one-sided), so the point is a typical
# hour, not an expectation; the P10-P90 band keeps the spike side.
MODEL_PARAMS = dict(DEFAULT_MODEL_PARAMS, loss="absolute_error")
OFFSET_COL = "dam_price"


def main(vintage="pre_dam", target_date=None, out_dir=DATA_DIR, dam_forecast_dir=DATA_DIR, backtest_enabled=True,
         zone=ZONE):
    rtm = load_price_series("ieso_rtm_prices.csv", zone)
    dam = load_price_series("ieso_dam_prices.csv", zone)
    return run_forecast("rtm", rtm, FEATURE_COLS, dam, attach_dam_feature=True, model_params=MODEL_PARAMS,
                        offset_col=OFFSET_COL, vintage=vintage,
                        target_date=target_date, out_dir=out_dir, dam_forecast_dir=dam_forecast_dir,
                        backtest_enabled=backtest_enabled, zone=zone)


if __name__ == "__main__":
    args = parse_run_args("Forecast tomorrow's RTM price.")
    main(args.vintage, args.target_date, args.out_dir, zone=args.zone)
