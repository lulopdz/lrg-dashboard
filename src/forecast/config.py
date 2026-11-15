"""Shared configuration, constants, zone helpers, and CLI parsing for the forecast pipeline.

Kept free of scikit-learn imports at module top level so multiprocessing callers such as
walkforward.py can apply threadpoolctl limits before OpenMP/scikit-learn initializes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
# Day-ahead vintages of the inputs (see vintages.py): rows for past target days as they
# looked at the 9:00 run before them. Wherever they exist they replace the freshest
# version, so training sees the same quality of input the model gets for tomorrow.
DAYAHEAD_PATH = DATA_DIR / "forecast_inputs_dayahead.csv"

ZONE = "OTTAWA"  # the dashboard's zone; its files carry no suffix (dam_forecast.csv), other zones do (dam_forecast_NIAGARA.csv)
ZONES = ("OTTAWA", "TORONTO", "NIAGARA", "NORTHWEST")  # what the workflows forecast every day (NORTHWEST: D-4, 2026-10-02)


# Inputs only one zone's models see, on top of each model's own FEATURE_COLS. NORTHWEST trades
# ~1,000 km from Ottawa and Toronto, so its own weather (Thunder Bay, update_weather.py) is the
# local demand signal there (TRACKING N-4). Its regional load forecast, ontario_northwest, is
# already in every model's FEATURE_COLS.
ZONE_FEATURES = {
    "NORTHWEST": ["temperature_2m_thunder_bay", "wind_speed_10m_thunder_bay", "relative_humidity_2m_thunder_bay"],
}


def zone_feature_cols(feature_cols, zone):
    return list(feature_cols) + [c for c in ZONE_FEATURES.get(zone, []) if c not in feature_cols]


def zone_suffix(zone):
    return "" if zone == ZONE else f"_{zone}"


def forecast_file(prefix, kind, zone=ZONE):
    """kind: 'forecast' (latest run), 'forecast_meta', 'forecast_history', 'signal',
    'signal_meta', 'signal_history'. E.g. forecast_file('rtm', 'forecast_history', 'NIAGARA')
    -> 'rtm_forecast_history_NIAGARA.csv'."""
    ext = "json" if kind.endswith("meta") else "csv"
    return f"{prefix}_{kind}{zone_suffix(zone)}.{ext}"


# IESO publishes its wind forecast per zone. 'Ontario Total' looks like the natural choice but
# only exists from 2026-06-27 -- 13% of the training history -- so the models were being fed a
# column that was NaN for most of what they trained on. 'West' covers the full history
# (2025-05-20 on), is equally complete for the target day, and is where the wind fleet actually
# sits (Port Alma, the secondary weather station, is in the same region). Walk-forward: DAM
# MAE $11.46 -> $10.86, RTM $34.05 -> $33.77.
WIND_ZONE = "West"
BACKTEST_DAYS = 21   # trailing holdout window used to report honest accuracy
MIN_TRAIN_DAYS = 60  # need enough history before the holdout window to bother training

# Two runs a day, each a "vintage" archived separately (scorecard.py scores them apart):
#   pre_dam  -- the 9:00 Ottawa run. Tomorrow's DAM is not out yet (IESO publishes ~12:35 EST),
#               so the DAM feature is predict_dam.py's own forecast. This is the vintage a
#               virtual bid at the 10:00 deadline can act on.
#   post_dam -- the afternoon re-run once tomorrow's DAM is published: same models, real DAM
#               feature, for anyone deciding real-time exposure.
# RT_HOURS_KNOWN is how many of *today's* RT hours are complete when each run happens (the
# 9:00 Ottawa run is 8:00 EST in summer, so hours 0-6 are in; the afternoon run sees 0-12).
# The walk-forward harness cuts history there so a reconstruction sees exactly what the live
# run saw; the same constant caps the same-day features in add_lag_features.
VINTAGES = ("pre_dam", "post_dam")
# The 2026-09-21 feature set (supply-side columns, is_holiday, RT lags from two days back plus
# today's settled hours, the P10-P90 band) against the one before it, walk-forward over the
# 68 archived days (2026-07-10..09-22, pre_dam): DAM MAE $9.03 -> $8.55, RTM $21.81 -> $22.72,
# Spread $22.26 -> $22.46; no difference clears a 90% bootstrap interval. Kept: DAM improves,
# and the supply columns target the spring SBG regime (RT ~$0), which a July-September window
# barely contains (7 such hours) -- re-check once spring 2027 is in the archive.
RT_HOURS_KNOWN = {"pre_dam": 7, "post_dam": 13}

# Default booster settings. ~11k training rows against ~40 features is small for
# unbounded-depth boosting, so capping depth is plain regularization; a 6-fold walk-forward
# showed it clearly helps the noisier RTM and Spread series (RTM MAE $34.05 -> $32.97, Spread
# $34.77 -> $32.73, each winning 4-5 of 6 folds) while leaving DAM flat ($11.46 -> $11.48).
# predict_dam.py therefore overrides this back to scikit-learn's defaults; see MODEL_PARAMS
# in each predict_*.py.
DEFAULT_MODEL_PARAMS = dict(max_depth=4, learning_rate=0.05, max_iter=300)

# backtest() returns the raw holdout series under these keys for directional_backtest() to
# score; run_forecast() drops them before writing the meta JSON (see the comment in backtest).
BACKTEST_SERIES_KEYS = ("test_actual", "test_predicted")

# The band around the point forecast: two more boosters on the pinball loss for these
# quantiles. Replaces the old symmetric +/- backtest-MAE band, which said nothing about which
# side the risk was on -- RT's distribution is one-sided (a long right tail of spikes and, in
# spring, a floor at $0), so the P10 and P90 sit at very different distances from the point.
QUANTILES = (0.1, 0.9)

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

# Supply-side inputs from IESO's adequacy report (update_adequacy.py). Only fields the early
# day-ahead vintage already carries at the 9:00 run (capacities, outages, IESO's own demand,
# wind and solar forecasts); the "scheduled" columns are filled in later and would be present
# in training but NaN at serve time. What they buy: net_load -- demand less nuclear less wind
# and solar, the load left for hydro and gas -- is the one number that separates the ~$0 SBG
# hours (RT <= $1 in 58% of hours below 1,000 MW vs 0.1% above 5,000 MW, 2025-05 to 2026-09),
# and none of it was visible to the models before 2026-09-21's $0 hours.
SUPPLY_COLS = ["nuclear_avail", "gas_avail", "hydro_avail", "adq_demand", "solar_forecast",
               "net_load", "net_load_day_min", "net_load_day_max", "capacity_margin"]

HISTORY_COLS = ["target_date", "vintage", "generated_at", "analog_date", "analog_date_2", "hour",
                "predicted_lmp", "p10", "p90", "analog_lmp", "analog_lmp_2", "band_err", "backfilled"]

# Where the money is lost: from 2026-09-24 OTTAWA's 10 worst DART hours were all HE17-19, the
# evening ramp -- load climbing while solar drops out, earlier every autumn day -- and HE17-20
# is where the picks land most. Nothing told the models where an hour sits against sunset, or
# how steep the ramp into and out of it is forecast to be. Added to the RT model and the spread
# signal 2026-10-31; full-history walk-forward (2025-07-19..2026-10-30, pre_dam), 5 picks a day:
# more P&L in all four zones, both on the 106 days with the 9:00 inputs (+$4.4 to +$13.3/day,
# 90% interval clear of zero in OTTAWA and NIAGARA) and on the rest (+$5.5 to +$18.0/day); RT
# MAE down $0.2-0.6 everywhere, still above RT = predicted DAM.
RAMP_COLS = ["hours_from_sunset", "load_ramp_in", "load_ramp_out", "net_load_ramp_in", "net_load_ramp_out", "solar_ramp_out"]
OTTAWA_LAT, OTTAWA_LON = 45.42, -75.70


def parse_run_args(description):
    """The command line shared by predict_dam.py / predict_rtm.py / predict_spread.py:
    which vintage to produce, and (for reconstructions) which past day and where to write."""
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--vintage", choices=VINTAGES, default="pre_dam",
                    help="pre_dam: the 9:00 run for tomorrow (default); post_dam: the afternoon re-run with tomorrow's real DAM")
    ap.add_argument("--target-date", default=None, help="YYYY-MM-DD: reconstruct a past day walk-forward instead of the live target")
    ap.add_argument("--out-dir", default=str(DATA_DIR), help="where to write outputs and the history archive (default: data/)")
    ap.add_argument("--zone", default=ZONE, help=f"virtual zonal hub to forecast (default {ZONE}; the dashboard's zone)")
    args = ap.parse_args()
    args.target_date = pd.Timestamp(args.target_date).date() if args.target_date else None
    return args
