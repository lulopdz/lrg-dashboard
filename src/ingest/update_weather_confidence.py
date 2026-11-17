"""Builds data/weather_confidence.csv: how much to trust the Open-Meteo forecast the Weather
tab displays.

Two independent signals, because they can disagree and the disagreement is informative:

  A. Ensemble spread -- ECMWF IFS runs 51 perturbed members. The p10..p90 range across them
     at a given hour is how much the models disagree *right now*. Wide band = genuinely
     uncertain hour.
  B. Run-to-run revision -- what the forecast for that same hour said 1/2/3 days ago
     (Open-Meteo's previous-runs endpoint). Large revisions = the forecast keeps changing its
     mind, which a tight ensemble band would not reveal on its own.

ECMWF IFS specifically: of the ensemble models Open-Meteo carries, it is the only one that
returns hub-height (100m) wind -- GFS, ICON and GEM all return member keys full of nulls for
wind_speed_80m/100m/120m. It also has the most members (51).

Ensemble history is short (the endpoint only serves a few past days), so this file accumulates:
each run appends the current window and older rows are kept. Run alongside update_weather.py.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "weather_confidence.csv"
DAYAHEAD_PATH = Path(__file__).resolve().parents[2] / "data" / "weather_confidence_dayahead.csv"

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs025"

OTTAWA = {"lat": 45.4000, "lon": -75.7000}
PORT_ALMA = {"lat": 42.1808, "lon": -82.2444}
TORONTO = {"lat": 43.6532, "lon": -79.3832}

# (request variable, column prefix used in the output + in OTTAWA_weather.csv)
OTTAWA_VARS = [
    ("temperature_2m", "temperature_2m"),
    ("relative_humidity_2m", "relative_humidity_2m"),
    ("precipitation", "precipitation"),
    ("snowfall", "snowfall"),
    ("wind_speed_10m", "wind_speed_10m"),
    ("shortwave_radiation", "shortwave_radiation"),
]
PORT_ALMA_VARS = [
    ("wind_speed_100m", "wind_speed_100m_port_alma"),
    ("shortwave_radiation", "shortwave_radiation_port_alma"),
]
TORONTO_VARS = [
    ("temperature_2m", "temperature_2m_toronto"),
    ("wind_speed_10m", "wind_speed_10m_toronto"),
    ("relative_humidity_2m", "relative_humidity_2m_toronto"),
]

PAST_DAYS = 3      # as far back as the ensemble endpoint serves
# Days are counted in UTC (timezone=UTC below), so the window ends at 23:00 UTC = 18:00 Ottawa:
# with 3, the last day lost HE20-24, and on a day the fetch failed that last day was tomorrow,
# so its band stopped at 18:00 (2026-11-18). 4 keeps tomorrow whole even with a day missed.
FORECAST_DAYS = 4
RETRIES = 3        # attempts per request; the endpoint failed 4 of 30 daily runs in Oct-Nov 2026
RETRY_WAIT_S = 20
REVISION_DAYS = (1, 2, 3)  # compare against the runs from 1, 2 and 3 days earlier


def _get(url, params):
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=180)
            r.raise_for_status()
            payload = r.json()
            break
        except (requests.RequestException, ValueError) as exc:
            if attempt == RETRIES:
                raise RuntimeError(f"{url}: request/JSON error after {RETRIES} attempts: {exc}") from exc
            print(f"{url}: attempt {attempt} failed ({exc}); retrying in {RETRY_WAIT_S}s")
            time.sleep(RETRY_WAIT_S)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{url}: unexpected response payload type {type(payload).__name__}")
    if payload.get("error"):
        raise RuntimeError(f"{url}: {payload.get('reason')}")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not hourly.get("time"):
        raise RuntimeError(f"{url}: missing hourly time series in response")
    return hourly


def _timestamps(hourly):
    return pd.to_datetime(hourly["time"], utc=True).tz_convert("-05:00")


def ensemble_spread(coords, variables):
    """p10 / p50 / p90 across the ensemble members, per hour, per variable."""
    hourly = _get(ENSEMBLE_URL, {
        "latitude": coords["lat"], "longitude": coords["lon"],
        "hourly": ",".join(v for v, _ in variables), "models": MODEL,
        "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
        "timezone": "UTC", "wind_speed_unit": "ms",
    })
    out = pd.DataFrame({"timestamp": _timestamps(hourly)})
    if out.empty:
        raise RuntimeError("Ensemble response returned 0 hours")
    for var, prefix in variables:
        members = [k for k in hourly if k == var or k.startswith(f"{var}_member")]
        if not members:
            raise RuntimeError(f"Ensemble response missing member series for {var}")
        # shape (hours, members); nanpercentile so a member dropping out doesn't void the hour
        block = np.array([hourly[k] for k in members], dtype=float).T
        with np.errstate(all="ignore"):
            valid = ~np.all(np.isnan(block), axis=1)
            p10 = np.full(len(block), np.nan)
            p50 = np.full(len(block), np.nan)
            p90 = np.full(len(block), np.nan)
            if valid.any():
                p10[valid] = np.nanpercentile(block[valid], 10, axis=1)
                p50[valid] = np.nanpercentile(block[valid], 50, axis=1)
                p90[valid] = np.nanpercentile(block[valid], 90, axis=1)
        out[f"{prefix}_p10"] = np.round(p10, 2)
        out[f"{prefix}_p50"] = np.round(p50, 2)
        out[f"{prefix}_p90"] = np.round(p90, 2)
        out[f"{prefix}_members"] = int(len(members))
    return out


def run_revisions(coords, variables):
    """What earlier model runs said for the same hour, as {prefix}_prev{N}."""
    requested = []
    for var, _ in variables:
        requested.append(var)
        requested += [f"{var}_previous_day{d}" for d in REVISION_DAYS]
    hourly = _get(PREVIOUS_RUNS_URL, {
        "latitude": coords["lat"], "longitude": coords["lon"],
        "hourly": ",".join(requested),
        "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
        "timezone": "UTC", "wind_speed_unit": "ms",
    })
    out = pd.DataFrame({"timestamp": _timestamps(hourly)})
    if out.empty:
        raise RuntimeError("Previous-runs response returned 0 hours")
    for var, prefix in variables:
        for d in REVISION_DAYS:
            key = f"{var}_previous_day{d}"
            if key in hourly:
                out[f"{prefix}_prev{d}"] = np.round(np.array(hourly[key], dtype=float), 2)
    return out


def build():
    frames = []
    for coords, variables, label in [(OTTAWA, OTTAWA_VARS, "Ottawa"),
                                      (PORT_ALMA, PORT_ALMA_VARS, "Port Alma"),
                                      (TORONTO, TORONTO_VARS, "Toronto")]:
        spread = ensemble_spread(coords, variables)
        revisions = run_revisions(coords, variables)
        merged = spread.merge(revisions, on="timestamp", how="outer")
        print(f"{label}: {len(merged)} hours, {len(merged.columns) - 1} columns")
        frames.append(merged.set_index("timestamp"))

    fresh = pd.concat(frames, axis=1).reset_index()
    if fresh.empty:
        raise RuntimeError("Weather confidence fetch produced no rows.")
    # When each row was fetched, so the Weather tab can say how old the band it draws is (a
    # failed run leaves the previous day's rows in place without any other sign).
    fresh["fetched_at"] = pd.Timestamp.now(tz="UTC").tz_convert("-05:00").strftime("%Y-%m-%d %H:%M")

    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        if "timestamp" not in existing.columns:
            raise ValueError(f"{DATA_PATH} is missing expected column: 'timestamp'")
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        combined = pd.concat([existing, fresh], axis=0, ignore_index=True) if not existing.empty else fresh
    else:
        combined = fresh

    # Newest run wins for an hour we've seen before -- the same convention the price and
    # weather feeds use, since a shorter lead time means a better forecast.
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(combined)} rows to {DATA_PATH} "
          f"(span {combined['timestamp'].min()} -> {combined['timestamp'].max()})")
    archive_dayahead(fresh)
    return combined


def archive_dayahead(fresh, path=DAYAHEAD_PATH, today=None):
    """Keep tomorrow's rows as fetched the day before. weather_confidence.csv overwrites each
    hour with the newest run, so for a past day it holds a short-lead ensemble, not what was
    known when the picks were made; this file is what lets the weather-context columns on the
    Spread Signal tab be re-tested against the picks later (TRACKING, 2026-11-17). Upserts
    tomorrow's rows (a later run the same day replaces them)."""
    ts = pd.to_datetime(fresh["timestamp"])
    today = today or pd.Timestamp.now(tz="UTC").tz_convert("-05:00").date()
    tomorrow = today + pd.Timedelta(days=1)
    rows = fresh[ts.dt.date == tomorrow].copy()
    if rows.empty:
        print(f"No rows for {tomorrow} in this fetch; day-ahead archive unchanged.")
        return None
    rows.insert(0, "target_date", str(tomorrow))
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["target_date"] != str(tomorrow)]
        if not old.empty:
            rows = pd.concat([old, rows], ignore_index=True)
    rows = rows.sort_values(["target_date", "timestamp"])
    rows.to_csv(path, index=False)
    print(f"Archived {str(tomorrow)} day-ahead ensemble to {path} ({rows['target_date'].nunique()} days)")
    return rows


if __name__ == "__main__":
    sys.exit(0 if build() is not None else 1)
