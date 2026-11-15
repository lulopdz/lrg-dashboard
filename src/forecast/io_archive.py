"""Input CSV loading/caching, day-ahead vintage overlay, target-date rules, and history archiving."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config import (
    DATA_DIR,
    DAYAHEAD_PATH,
    HISTORY_COLS,
    RT_HOURS_KNOWN,
    WIND_ZONE,
    ZONE,
    forecast_file,
)

_CSV_CACHE = {}
_DEFAULT_DAYAHEAD_PATH = DAYAHEAD_PATH


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


def _resolve_dayahead_path():
    # Support monkeypatching DAYAHEAD_PATH on io_archive, config, or the forecast_common facade.
    if DAYAHEAD_PATH != _DEFAULT_DAYAHEAD_PATH:
        return DAYAHEAD_PATH
    if config.DAYAHEAD_PATH != _DEFAULT_DAYAHEAD_PATH:
        return config.DAYAHEAD_PATH
    fc = sys.modules.get("forecast_common")
    if fc is not None:
        fc_path = getattr(fc, "DAYAHEAD_PATH", _DEFAULT_DAYAHEAD_PATH)
        if fc_path != _DEFAULT_DAYAHEAD_PATH:
            return fc_path
    return DAYAHEAD_PATH


def load_dayahead_inputs():
    path = _resolve_dayahead_path()
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["interval_start_local"])


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


def upsert_history_csv(path, rows, target_date, vintage):
    """Upsert one (target_date, vintage) row set into a history archive CSV, keeping rows
    sorted by (target_date, vintage, hour). Returns the full sorted DataFrame."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        if "vintage" not in old.columns:  # archives written before vintages existed
            old["vintage"] = "pre_dam"
        if "backfilled" not in old.columns and "backfilled" in rows.columns:
            old["backfilled"] = False  # archives written before the flag: all live runs
        keep = ~((old["target_date"] == target_date) & (old["vintage"] == vintage))
        rows = pd.concat([old[keep], rows])
    rows = rows.sort_values(["target_date", "vintage", "hour"])
    rows.to_csv(path, index=False)
    return rows


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
    # A walk-forward reconstruction (walkforward.py) is marked so scorecard.py grades it apart
    # from what was actually published -- it sees the latest inputs, not the 9:00 ones.
    rows["backfilled"] = bool(meta.get("reconstruction"))
    rows = rows[HISTORY_COLS]
    path = Path(out_dir) / forecast_file(prefix, "forecast_history", zone)
    rows = upsert_history_csv(path, rows, meta["target_date"], meta["vintage"])
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
    # The latest run only if it is for this day: a reconstruction of a day the archive doesn't
    # cover would otherwise be fed tomorrow's DAM forecast as its own.
    latest = source_dir / forecast_file("dam", "forecast", zone)
    meta = source_dir / forecast_file("dam", "forecast_meta", zone)
    if latest.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("target_date") == str(target_date):
        f = pd.read_csv(latest)
        return dict(zip(f["hour"], f["predicted_lmp"]))
    return {}
