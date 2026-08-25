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
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "weather_confidence.csv"

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs025"

OTTAWA = {"lat": 45.4000, "lon": -75.7000}
PORT_ALMA = {"lat": 42.1808, "lon": -82.2444}

# (request variable, column prefix used in the output + in OTTAWA_weather.csv)
OTTAWA_VARS = [
    ("temperature_2m", "temperature_2m"),
    ("relative_humidity_2m", "relative_humidity_2m"),
    ("precipitation", "precipitation"),
    ("snowfall", "snowfall"),
    ("wind_speed_10m", "wind_speed_10m"),
    ("shortwave_radiation", "shortwave_radiation"),
]
PORT_ALMA_VARS = [("wind_speed_100m", "wind_speed_100m_port_alma")]

PAST_DAYS = 3      # as far back as the ensemble endpoint serves
FORECAST_DAYS = 3
REVISION_DAYS = (1, 2, 3)  # compare against the runs from 1, 2 and 3 days earlier


def _get(url, params):
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"{url}: {payload.get('reason')}")
    return payload["hourly"]


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
    for var, prefix in variables:
        members = [k for k in hourly if k == var or k.startswith(f"{var}_member")]
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
    for var, prefix in variables:
        for d in REVISION_DAYS:
            key = f"{var}_previous_day{d}"
            if key in hourly:
                out[f"{prefix}_prev{d}"] = np.round(np.array(hourly[key], dtype=float), 2)
    return out


def build():
    frames = []
    for coords, variables, label in [(OTTAWA, OTTAWA_VARS, "Ottawa"),
                                      (PORT_ALMA, PORT_ALMA_VARS, "Port Alma")]:
        spread = ensemble_spread(coords, variables)
        revisions = run_revisions(coords, variables)
        merged = spread.merge(revisions, on="timestamp", how="outer")
        print(f"{label}: {len(merged)} hours, {len(merged.columns) - 1} columns")
        frames.append(merged.set_index("timestamp"))

    fresh = pd.concat(frames, axis=1).reset_index()

    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        combined = pd.concat([existing, fresh], axis=0)
    else:
        combined = fresh

    # Newest run wins for an hour we've seen before -- the same convention the price and
    # weather feeds use, since a shorter lead time means a better forecast.
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(combined)} rows to {DATA_PATH} "
          f"(span {combined['timestamp'].min()} -> {combined['timestamp'].max()})")
    return combined


if __name__ == "__main__":
    sys.exit(0 if build() is not None else 1)
