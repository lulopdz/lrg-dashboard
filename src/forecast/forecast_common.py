"""Shared plumbing for the next-day DAM/RTM/Spread predictors (predict_dam.py, predict_rtm.py,
predict_spread.py): loading IESO's own forecasts, calendar feature engineering, backtesting,
the similar-day analog search, and the end-to-end run_forecast() driver they all call."""
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ZONE = "OTTAWA"  # the dashboard's zone; its files carry no suffix (dam_forecast.csv), other zones do (dam_forecast_NIAGARA.csv)
ZONES = ("OTTAWA", "TORONTO", "NIAGARA")  # what the workflows forecast every day


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


def parse_run_args(description):
    """The command line shared by predict_dam.py / predict_rtm.py / predict_spread.py:
    which vintage to produce, and (for reconstructions) which past day and where to write."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--vintage", choices=VINTAGES, default="pre_dam",
                    help="pre_dam: the 9:00 run for tomorrow (default); post_dam: the afternoon re-run with tomorrow's real DAM")
    ap.add_argument("--target-date", default=None, help="YYYY-MM-DD: reconstruct a past day walk-forward instead of the live target")
    ap.add_argument("--out-dir", default=str(DATA_DIR), help="where to write outputs and the history archive (default: data/)")
    ap.add_argument("--zone", default=ZONE, help=f"virtual zonal hub to forecast (default {ZONE}; the dashboard's zone)")
    args = ap.parse_args()
    args.target_date = pd.Timestamp(args.target_date).date() if args.target_date else None
    return args


_CSV_CACHE = {}


def read_input_csv(path, **kwargs):
    """pd.read_csv for the input feeds, memoised on (path, mtime, kwargs) and handed back as a
    copy. One process forecasting three zones x three models (run_forecasts.py), or rebuilding
    68 days (walkforward.py), otherwise parses the same ~20 MB of CSVs dozens of times; the
    mtime in the key means a file rewritten mid-process is read again. Only for files nothing
    writes during a forecast run -- not the forecast archives or the day-ahead inputs."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns, repr(sorted(kwargs.items())))
    if key not in _CSV_CACHE:
        _CSV_CACHE[key] = pd.read_csv(path, **kwargs)
    return _CSV_CACHE[key].copy()


def load_price_series(filename, zone=ZONE):
    """DAM or RTM hourly price for one zone -- both CSVs share the same
    interval_start_local / location / lmp schema."""
    df = read_input_csv(DATA_DIR / filename, parse_dates=["interval_start_local"])
    df = df[df["location"] == zone][["interval_start_local", "lmp"]].sort_values("interval_start_local")
    return df


def load_forecast_inputs():
    """The exogenous forecasts available for tomorrow, common to both predictors."""
    load_cols = ["ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast"]
    load_fc = read_input_csv(DATA_DIR / "ieso_load_forecast.csv", parse_dates=["interval_start_local"])
    load_fc = load_fc[["interval_start_local"] + load_cols].drop_duplicates("interval_start_local", keep="last")

    wind_fc = read_input_csv(DATA_DIR / "ieso_wind_forecast.csv", parse_dates=["interval_start_local"])
    wind_fc = wind_fc[wind_fc["zone"] == WIND_ZONE][["interval_start_local", "generation_forecast"]]
    wind_fc = wind_fc.rename(columns={"generation_forecast": "wind_forecast"}).drop_duplicates("interval_start_local", keep="last")

    weather = read_input_csv(DATA_DIR / "OTTAWA_weather.csv", parse_dates=["timestamp"])
    weather = weather.rename(columns={"timestamp": "interval_start_local"}).drop_duplicates("interval_start_local", keep="last")

    return load_fc, wind_fc, weather


# Day-ahead vintages of the inputs (see vintages.py): rows for past target days as they
# looked at the 9:00 run before them. Wherever they exist they replace the freshest
# version, so training sees the same quality of input the model gets for tomorrow.
DAYAHEAD_PATH = DATA_DIR / "forecast_inputs_dayahead.csv"


def load_dayahead_inputs():
    if not DAYAHEAD_PATH.exists():
        return None
    return pd.read_csv(DAYAHEAD_PATH, parse_dates=["interval_start_local"])


def overlay_dayahead(df):
    da = load_dayahead_inputs()
    if da is None or da.empty:
        return df
    cols = [c for c in da.columns if c in df.columns and c not in ("interval_start_local", "target_date")]
    da = da.drop_duplicates("interval_start_local").set_index("interval_start_local")[cols]
    idx = df["interval_start_local"]
    hit = idx.isin(da.index)
    if hit.any():
        current = df.loc[hit, cols]
        vintage = pd.DataFrame(da.loc[idx[hit]].values, index=current.index, columns=cols)
        # A column the archived commit didn't have yet (adequacy before 2026-08-25, the Port
        # Alma station) stays as it is rather than being blanked.
        df.loc[hit, cols] = vintage.where(vintage.notna(), current)
    return df


# Supply-side inputs from IESO's adequacy report (update_adequacy.py). Only fields the early
# day-ahead vintage already carries at the 9:00 run (capacities, outages, IESO's own demand,
# wind and solar forecasts); the "scheduled" columns are filled in later and would be present
# in training but NaN at serve time. What they buy: net_load -- demand less nuclear less wind
# and solar, the load left for hydro and gas -- is the one number that separates the ~$0 SBG
# hours (RT <= $1 in 58% of hours below 1,000 MW vs 0.1% above 5,000 MW, 2025-05 to 2026-09),
# and none of it was visible to the models before 2026-09-21's $0 hours.
SUPPLY_COLS = ["nuclear_avail", "gas_avail", "hydro_avail", "adq_demand", "solar_forecast",
               "net_load", "net_load_day_min", "net_load_day_max", "capacity_margin"]


def load_supply_inputs(source=None):
    """source: a path or file-like; default data/ieso_adequacy.csv. None if it doesn't exist."""
    path = DATA_DIR / "ieso_adequacy.csv" if source is None else source
    if isinstance(path, Path) and not path.exists():
        return None
    cols = ["interval_start_local", "ontario_demand_forecast", "nuclear_capacity", "nuclear_outages",
            "gas_capacity", "gas_outages", "hydro_capacity", "hydro_outages", "wind_forecasted",
            "solar_forecasted", "capacity_excess_shortfall"]
    reader = read_input_csv if isinstance(path, Path) else pd.read_csv  # vintages.py passes a StringIO
    adq = reader(path, usecols=cols, parse_dates=["interval_start_local"])
    for c in cols[1:]:
        adq[c] = pd.to_numeric(adq[c], errors="coerce")
    adq = adq.drop_duplicates("interval_start_local", keep="last")
    out = pd.DataFrame({"interval_start_local": adq["interval_start_local"]})
    out["nuclear_avail"] = adq["nuclear_capacity"] - adq["nuclear_outages"]
    out["gas_avail"] = adq["gas_capacity"] - adq["gas_outages"]
    out["hydro_avail"] = adq["hydro_capacity"] - adq["hydro_outages"]
    out["adq_demand"] = adq["ontario_demand_forecast"]
    out["solar_forecast"] = adq["solar_forecasted"].fillna(0)
    out["net_load"] = out["adq_demand"] - out["nuclear_avail"] - adq["wind_forecasted"].fillna(0) - out["solar_forecast"]
    day = out["interval_start_local"].dt.normalize()
    out["net_load_day_min"] = day.map(out.groupby(day)["net_load"].min())
    out["net_load_day_max"] = day.map(out.groupby(day)["net_load"].max())
    out["capacity_margin"] = adq["capacity_excess_shortfall"]
    return out


HISTORY_COLS = ["target_date", "vintage", "generated_at", "analog_date", "analog_date_2", "hour",
                "predicted_lmp", "p10", "p90", "analog_lmp", "analog_lmp_2", "band_err"]


def archive_forecast(prefix, out, meta, out_dir=DATA_DIR, zone=ZONE):
    """Append this run to {out_dir}/{prefix}_forecast_history.csv, one row set per
    (target_date, vintage) -- a later run of the same vintage for the same day replaces the
    earlier one. {prefix}_forecast.csv only ever holds the latest run; the archive is what
    lets the dashboard show past forecasts and scorecard.py grade them."""
    # Keys are ints straight out of backtest() but strings once the meta has been through JSON
    # (backfill_history.py); looking up only str(h) left band_err empty from 2026-09-18 on.
    hourly_mae = {int(h): v for h, v in ((meta.get("backtest") or {}).get("hourly_mae") or {}).items()}
    rows = out.copy()
    rows["band_err"] = rows["hour"].map(lambda h: hourly_mae.get(int(h))).astype(float).round(2)
    for col in ("target_date", "vintage", "generated_at", "analog_date", "analog_date_2"):
        rows[col] = meta.get(col)
    for col in ("p10", "p90"):
        if col not in rows.columns:
            rows[col] = np.nan
    rows = rows[HISTORY_COLS]
    path = Path(out_dir) / forecast_file(prefix, "forecast_history", zone)
    if path.exists():
        old = pd.read_csv(path)
        if "vintage" not in old.columns:  # archives written before vintages existed
            old["vintage"] = "pre_dam"
        keep = ~((old["target_date"] == meta["target_date"]) & (old["vintage"] == meta["vintage"]))
        rows = pd.concat([old[keep], rows])
    rows = rows.sort_values(["target_date", "vintage", "hour"])
    rows.to_csv(path, index=False)
    print(f"Archived {meta['target_date']} ({meta['vintage']}) to {path} ({rows['target_date'].nunique()} days)")


class TargetDateError(RuntimeError):
    """The DAM file doesn't put a live run's target on tomorrow (see determine_target_date).
    dam_already_out: tomorrow's DAM is in, so a pre_dam run has nothing honest left to make."""

    def __init__(self, message, dam_already_out=False):
        super().__init__(message)
        self.dam_already_out = dam_already_out


def market_today():
    return pd.Timestamp.now(tz="UTC").tz_convert("-05:00").date()  # IESO market time: EST all year


def determine_target_date(dam, vintage="pre_dam", today=None):
    """The day being forecast, anchored on DAM (published as a full day at once) so the DAM
    and RTM predictors always target the same day -- otherwise a predicted spread wouldn't
    line up. pre_dam: the day after the last published DAM (tomorrow, at the 9:00 run).
    post_dam: the last published DAM day itself, i.e. tomorrow once the afternoon fetch has
    brought its DAM in.

    A live run always forecasts tomorrow (market time), so anything else raises
    TargetDateError instead of quietly archiving the wrong day: a pre_dam run launched after
    tomorrow's DAM is out would otherwise target the day after (inputs half missing, archived
    as pre_dam), and one run while today's DAM is missing would target today."""
    last = dam["interval_start_local"].max().normalize().date()
    target = last if vintage == "post_dam" else last + pd.Timedelta(days=1)
    tomorrow = (today or market_today()) + pd.Timedelta(days=1)
    if target != tomorrow:
        if target > tomorrow:
            hint = ("tomorrow's DAM is already published, so there is no pre_dam forecast left to make "
                    "for it -- run --vintage post_dam")
        else:
            hint = "the DAM file is behind -- fetch the missing DAM day first (update_prices_ieso.py dam)"
        raise TargetDateError(f"{vintage} run would target {target}, expected tomorrow {tomorrow} "
                              f"(last DAM day in file: {last}): {hint}", dam_already_out=target > tomorrow)
    return target


def information_cutoff(target_date, vintage, tz):
    """The last instant of RT data the run can see: today (target - 1) up to the hour the
    run happens (see RT_HOURS_KNOWN). Used to cut history in walk-forward reconstructions."""
    today = pd.Timestamp(target_date, tz=tz) - pd.Timedelta(days=1)
    return today + pd.Timedelta(hours=RT_HOURS_KNOWN[vintage])


def build_grid(target_df, load_fc, wind_fc, weather, target_date, tz, supply=None):
    """One continuous hourly grid spanning history through target_date, with the target
    series ('lmp'), forecast inputs, supply-side inputs and calendar features merged in.
    'lmp' is NaN for target_date -- that's the row(s) to predict."""
    start = target_df["interval_start_local"].min()
    end = pd.Timestamp(target_date, tz=tz) + pd.Timedelta(hours=23)
    idx = pd.date_range(start, end, freq="h", tz=tz)

    df = pd.DataFrame({"interval_start_local": idx})
    df = df.merge(target_df, on="interval_start_local", how="left")
    df = df.merge(load_fc, on="interval_start_local", how="left")
    df = df.merge(wind_fc, on="interval_start_local", how="left")
    df = df.merge(weather, on="interval_start_local", how="left")
    if supply is None:
        supply = load_supply_inputs()
    if supply is not None:
        df = df.merge(supply, on="interval_start_local", how="left")
    df = overlay_dayahead(df)

    df["hour"] = df["interval_start_local"].dt.hour
    df["dow"] = df["interval_start_local"].dt.dayofweek
    df["month"] = df["interval_start_local"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Ontario statutory holidays trade like a Sunday (Labour Day 2026 cleared ~$15 below the
    # Monday before it); is_weekend alone can't tell the model that.
    dates = df["interval_start_local"].dt.date
    on_holidays = holidays.CA(subdiv="ON", years=range(dates.min().year, dates.max().year + 1))
    df["is_holiday"] = dates.map(lambda d: d in on_holidays).astype(int)
    for col, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    return df


def add_lag_features(df, prefix, known_through_yesterday=False, vintage="pre_dam"):
    """Causal lags on df['lmp'], shifted by whole days on the complete hourly grid so nothing
    at hour t ever sees data from t or later -- and, just as important, nothing the run
    doesn't have yet when it happens.

    known_through_yesterday=True (DAM: published a full day at once, so the whole of today
    is known at the 9:00 run) keeps the same-hour lag at 1 day. False (RT and the spread:
    today's hours are still being settled) shifts every lag one more day, and instead
    summarises today so far -- the mean and max of the RT_HOURS_KNOWN[vintage] - 1 hours
    that are complete by run time -- as {prefix}_today_early_mean/max, applied to every hour
    of the target day. Before this, rtm_lag_1d was NaN for ~16 of the 24 target hours at
    serve time and NaN in 0.2% of training rows: the booster answered those hours from a
    branch it had never trained."""
    lmp = df["lmp"]
    shift_days = 1 if known_through_yesterday else 2
    df[f"{prefix}_lag_{shift_days}d"] = lmp.shift(24 * shift_days)
    df[f"{prefix}_lag_7d"] = lmp.shift(24 * 7)
    df[f"{prefix}_roll_7d"] = lmp.shift(24 * shift_days).rolling(24 * 7, min_periods=24 * 3).mean()
    df[f"{prefix}_roll_28d"] = lmp.shift(24 * shift_days).rolling(24 * 28, min_periods=24 * 7).mean()
    if not known_through_yesterday:
        # One hour short of RT_HOURS_KNOWN on purpose: that constant is the summer case (the
        # 9:00 Ottawa run is 8:00 EST, so hour 6 closed only an hour earlier), and a late RT
        # publication or an early trigger must not leave the feature computed from a different
        # set of hours live than in training. In winter (9:00 EST) the margin is two hours.
        early_hours = RT_HOURS_KNOWN[vintage] - 1
        day = df["interval_start_local"].dt.normalize()
        early = df[df["hour"] < early_hours].groupby(day)["lmp"].agg(["mean", "max"])
        prev_day = day - pd.Timedelta(days=1)
        df[f"{prefix}_today_early_mean"] = prev_day.map(early["mean"])
        df[f"{prefix}_today_early_max"] = prev_day.map(early["max"])
    return df


def usable_feature_cols(df, feature_cols):
    """Drop any feature that's entirely missing in this slice. HistGradientBoostingRegressor
    handles partial missingness fine, but a column with zero non-null values can crash its
    histogram binning step on some scikit-learn versions -- e.g. a forecast source (like the
    Ontario-wide wind forecast) that only has a couple weeks of history won't have a single
    real value in an older training window, even though it's fully populated more recently."""
    return [c for c in feature_cols if df[c].notna().any()]


def backtest(df_hist, feature_cols, naive_col, model_params=None):
    """Trailing holdout: honest accuracy vs. a naive 'same hour last week' baseline, plus
    a per-hour-of-day error breakdown (which hours the model has historically nailed vs.
    missed) used to recommend the hour we're most confident in. model_params must match what
    fit_final_model uses, or the reported accuracy won't describe the shipped model."""
    cutoff = df_hist["interval_start_local"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    train = df_hist[df_hist["interval_start_local"] < cutoff]
    test = df_hist[df_hist["interval_start_local"] >= cutoff]
    if len(train) < MIN_TRAIN_DAYS * 24 or test.empty:
        return None

    usable_cols = usable_feature_cols(train, feature_cols)
    model = HistGradientBoostingRegressor(random_state=0, **(model_params or {}))
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


# The band around the point forecast: two more boosters on the pinball loss for these
# quantiles. Replaces the old symmetric +/- backtest-MAE band, which said nothing about which
# side the risk was on -- RT's distribution is one-sided (a long right tail of spikes and, in
# spring, a floor at $0), so the P10 and P90 sit at very different distances from the point.
QUANTILES = (0.1, 0.9)


def order_band(df_target):
    """P10 <= point <= P90. The three boosters are fit independently, so nothing stops a
    quantile from landing on the wrong side of the point forecast (or of each other) in an
    hour with little training support; a band that doesn't contain its own centre reads as a
    bug on the chart."""
    lo = df_target[["p10", "p90", "predicted_lmp"]].min(axis=1)
    hi = df_target[["p10", "p90", "predicted_lmp"]].max(axis=1)
    return df_target.assign(p10=lo, p90=hi)


def missing_inputs(df_target, feature_cols):
    """(hours with any missing input, {column: missing hours}) for the target day. The
    booster still predicts through NaN, but along a branch trained on few rows -- e.g. the
    same-day RT summary when the 9:00 RT download failed -- so the meta records which
    inputs were missing and the page says so."""
    na = df_target[feature_cols].isna()
    n_hours = int(na.any(axis=1).sum())
    cols = {c: int(v) for c, v in na.sum().items() if v}
    if n_hours:
        print(f"Warning: {n_hours} of {len(df_target)} target hours have missing inputs "
              f"({', '.join(f'{c} x{v}' for c, v in cols.items())}) -- predictions for those hours are less reliable.")
    return n_hours, cols


def fit_quantile_model(df_hist, usable_cols, quantile, model_params=None):
    params = {k: v for k, v in (model_params or {}).items() if k != "loss"}
    model = HistGradientBoostingRegressor(random_state=0, loss="quantile", quantile=quantile, **params)
    model.fit(df_hist[usable_cols], df_hist["lmp"])
    return model


def fit_final_model(df_hist, feature_cols, model_params=None):
    """Returns (model, usable_cols) -- usable_cols is feature_cols minus anything entirely
    missing in df_hist; predict() calls must select the same columns, not the original list."""
    usable_cols = usable_feature_cols(df_hist, feature_cols)
    model = HistGradientBoostingRegressor(random_state=0, **(model_params or {}))
    model.fit(df_hist[usable_cols], df_hist["lmp"])
    return model, usable_cols


def attach_reference_price(df, reference_df, target_date, target_pred=None, feature_name="dam_price"):
    """Same-hour reference price as a feature (e.g. DAM price, for the RTM/Spread models):
    the actual published price for history, and -- when the actual price doesn't exist yet
    for target_date -- the matching predict_*.py script's prediction, passed as target_pred
    ({hour 1-24: value}). None means "use whatever the reference series has" (post_dam runs,
    where tomorrow's DAM is real). Returns (df, used_forecast).

    History keeps the real DAM even though a pre_dam run is served a forecast one. Training on
    out-of-fold DAM forecasts instead (month by month, a DAM model fit on earlier data) was
    tried on 2026-09-22: walk-forward RTM MAE $22.72 -> $23.60 and the spread signal's rated
    1 MW P&L $2,093 -> $258 over 67 days, so it was dropped."""
    df = df.merge(reference_df.rename(columns={"lmp": feature_name}), on="interval_start_local", how="left")
    target_mask = df["interval_start_local"].dt.date == target_date
    if target_pred is not None:
        df.loc[target_mask, feature_name] = (df.loc[target_mask, "hour"] + 1).map(target_pred)
    return df, target_pred is not None


def dam_target_prediction(target_date, vintage, source_dir=DATA_DIR, zone=ZONE):
    """The DAM prediction to use as tomorrow's dam_price feature. Live pre_dam runs read the
    fresh dam_forecast.csv; walk-forward reconstructions read the archived vintage for that
    day so a rebuilt RTM/Spread forecast is fed exactly what the live one was. post_dam runs
    get None: the real DAM is in the series already. Empty dict if nothing exists."""
    if vintage == "post_dam":
        return None
    source_dir = Path(source_dir)
    hist = source_dir / forecast_file("dam", "forecast_history", zone)
    if hist.exists():
        h = pd.read_csv(hist)
        if "vintage" in h.columns:
            h = h[h["vintage"] == "pre_dam"]
        h = h[h["target_date"] == str(target_date)]
        if len(h):
            return dict(zip(h["hour"], h["predicted_lmp"]))
    latest = source_dir / forecast_file("dam", "forecast", zone)
    if latest.exists():
        f = pd.read_csv(latest)
        return dict(zip(f["hour"], f["predicted_lmp"]))
    return {}


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


def run_forecast(prefix, series_df, feature_cols, dam, attach_dam_feature=False,
                 model_params=None, vintage="pre_dam", target_date=None, out_dir=DATA_DIR,
                 dam_forecast_dir=DATA_DIR, backtest_enabled=True, zone=ZONE):
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

    metrics = (backtest(df_hist, feature_cols, naive_col=f"{prefix}_lag_7d", model_params=model_params)
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

    model, usable_cols = fit_final_model(df_hist, feature_cols, model_params=model_params)
    missing_features, missing_cols = missing_inputs(df_target, feature_cols)
    df_target["predicted_lmp"] = model.predict(df_target[usable_cols])
    for q in QUANTILES:
        df_target[f"p{int(q * 100)}"] = fit_quantile_model(df_hist, usable_cols, q, model_params).predict(df_target[usable_cols])
    df_target = order_band(df_target)

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
        "zone": zone,
        "target_date": str(target_date),
        "vintage": vintage,
        "reconstruction": reconstruction,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "missing_input_hours": int(missing_features),
        "missing_input_cols": missing_cols,
        "analog_date": str(analogs[0]["date"]) if analogs else None,
        "analog_distance": analogs[0]["distance"] if analogs else None,
        "analog_comparison": comparison_display(0),
        "analog_date_2": str(analogs[1]["date"]) if len(analogs) > 1 else None,
        "analog_distance_2": analogs[1]["distance"] if len(analogs) > 1 else None,
        "analog_comparison_2": comparison_display(1),
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
