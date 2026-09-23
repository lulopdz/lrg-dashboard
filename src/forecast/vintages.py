"""Day-ahead vintages of the forecast inputs: the load, wind, weather and adequacy rows for a
target day *as they looked at the 9:00 run the day before*.

Why this exists: the ingest scripts keep only the freshest version of every hour (a forecast
published 2 h before delivery overwrites the one published 26 h before), so the history the
models train on is far more accurate than what they are given for tomorrow. Training on the
day-ahead vintage where it exists closes that gap, and the walk-forward harness needs the
same rows to reconstruct honestly.

Two sources, one file (data/forecast_inputs_dayahead.csv):
  - going forward, run_forecast() archives the target day's inputs on every live pre_dam run;
  - the past is rebuilt from git: every daily run committed the input CSVs right after
    forecasting, so the first commit after each archived forecast's generated_at holds
    exactly the inputs that run saw (`python src/forecast/vintages.py --rebuild`)."""
import argparse
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_common import DATA_DIR, ROOT, WIND_ZONE, load_supply_inputs  # noqa: E402

PATH = DATA_DIR / "forecast_inputs_dayahead.csv"
LOAD_COLS = ["ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast"]
INPUT_FILES = {
    "load": "data/ieso_load_forecast.csv",
    "wind": "data/ieso_wind_forecast.csv",
    "weather": "data/OTTAWA_weather.csv",
    "adequacy": "data/ieso_adequacy.csv",
}


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=ROOT).stdout


def _commits(path):
    """[(sha, commit time UTC)] touching path, oldest first."""
    out = []
    for line in _git("log", "--reverse", "--format=%H %cI", "--", path).splitlines():
        sha, ts = line.split()
        out.append((sha, pd.Timestamp(ts).tz_convert("UTC")))
    return out


def _read_at(sha, path, **kw):
    try:
        text = _git("show", f"{sha}:{path}")
    except subprocess.CalledProcessError:
        return None
    return pd.read_csv(io.StringIO(text), **kw)


def inputs_for_day(target_date, load_fc, wind_fc, weather, supply):
    """One frame with every input column for target_date's 24 hours, from in-memory inputs
    shaped like forecast_common.load_forecast_inputs() / load_supply_inputs() return them."""
    def day(df):
        if df is None:
            return None
        return df[df["interval_start_local"].dt.date == target_date]
    frames = [f for f in (day(load_fc), day(wind_fc), day(weather), day(supply)) if f is not None and len(f)]
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="interval_start_local", how="outer")
    out.insert(1, "target_date", str(target_date))
    return out.sort_values("interval_start_local")


def inputs_at_commit(sha, target_date):
    """The inputs for target_date as committed in sha, using the same column selection the
    live loaders apply."""
    load = _read_at(sha, INPUT_FILES["load"], parse_dates=["interval_start_local"])
    wind = _read_at(sha, INPUT_FILES["wind"], parse_dates=["interval_start_local"])
    weather = _read_at(sha, INPUT_FILES["weather"], parse_dates=["timestamp"])
    adequacy_text = None
    try:
        adequacy_text = _git("show", f"{sha}:{INPUT_FILES['adequacy']}")
    except subprocess.CalledProcessError:
        pass
    if load is not None:
        load = load[["interval_start_local"] + LOAD_COLS].drop_duplicates("interval_start_local", keep="last")
    if wind is not None:
        wind = wind[wind["zone"] == WIND_ZONE][["interval_start_local", "generation_forecast"]]
        wind = wind.rename(columns={"generation_forecast": "wind_forecast"}).drop_duplicates("interval_start_local", keep="last")
    if weather is not None:
        weather = weather.rename(columns={"timestamp": "interval_start_local"}).drop_duplicates("interval_start_local", keep="last")
    supply = load_supply_inputs(io.StringIO(adequacy_text)) if adequacy_text else None
    return inputs_for_day(target_date, load, wind, weather, supply)


CORE_COLS = ["ontario", "wind_forecast", "temperature_2m"]  # a vintage missing these is not worth keeping


def archive(rows):
    """Append rows (from inputs_for_day) to PATH; the first vintage archived for a target day
    is the one kept, since it is the earliest -- and honest -- view. A day whose inputs are
    incomplete (a feed that hadn't published yet, or was down) is skipped so a later, complete
    run can archive it."""
    if rows is None or rows.empty:
        return
    core = [c for c in CORE_COLS if c in rows.columns]
    if len(rows) < 24 or len(core) < len(CORE_COLS) or rows[core].isna().any().any():
        print(f"Day-ahead inputs for {rows['target_date'].iloc[0]} incomplete -- not archived")
        return
    if PATH.exists():
        old = pd.read_csv(PATH, parse_dates=["interval_start_local"])
        rows = pd.concat([old, rows[~rows["target_date"].isin(old["target_date"])]])
    rows.sort_values("interval_start_local").to_csv(PATH, index=False)


def load_dayahead_inputs():
    if not PATH.exists():
        return None
    return pd.read_csv(PATH, parse_dates=["interval_start_local"])


def rebuild_from_git():
    """One-off: for every archived pre_dam DAM forecast, the inputs as of the first commit
    after it ran (same day), else the last commit before it."""
    hist = pd.read_csv(DATA_DIR / "dam_forecast_history.csv")
    if "vintage" in hist.columns:
        hist = hist[hist["vintage"] == "pre_dam"]
    runs = hist.groupby("target_date")["generated_at"].first()
    commits = _commits(INPUT_FILES["load"])
    PATH.unlink(missing_ok=True)
    kept = 0
    for target, generated in runs.items():
        generated = pd.Timestamp(generated).tz_convert("UTC")
        after = [c for c in commits if c[1] >= generated and c[1] - generated < pd.Timedelta(hours=6)]
        before = [c for c in commits if c[1] < generated]
        pick = after[0] if after else (before[-1] if before else None)
        if pick is None:
            continue
        rows = inputs_at_commit(pick[0], pd.Timestamp(target).date())
        if rows is None:
            continue
        archive(rows)
        kept += 1
        print(f"{target}: inputs from {pick[0][:7]} ({pick[1]:%Y-%m-%d %H:%M} UTC, run {generated:%H:%M} UTC) -- {len(rows)} h")
    print(f"Archived day-ahead inputs for {kept} target days -> {PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Day-ahead vintages of the forecast inputs.")
    ap.add_argument("--rebuild", action="store_true", help="rebuild data/forecast_inputs_dayahead.csv from git history")
    args = ap.parse_args()
    if args.rebuild:
        rebuild_from_git()
    else:
        ap.print_help()
