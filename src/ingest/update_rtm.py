import argparse
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from gridstatusio import GridStatusClient

DATASET_ID = "ieso_lmp_real_time_5_min_virtual_zonal"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ieso_rtm_prices.csv"
DEFAULT_LOOKBACK_HOURS = 24  # used only when there's no existing file to anchor from
SAFETY_BUFFER_HOURS = 2  # re-fetch a small overlap in case IESO revises recent values
INTERVALS_PER_HOUR = 12  # 5-min intervals in a settled hour


REQUIRED_COLS = ["location", "interval_start_local", "lmp"]


def determine_start(now, lookback_hours):
    if lookback_hours is not None:
        if lookback_hours <= 0:
            raise ValueError(f"lookback_hours must be positive, got {lookback_hours}")
        return now - timedelta(hours=lookback_hours)

    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH)
        missing = [c for c in REQUIRED_COLS if c not in existing.columns]
        if missing:
            raise ValueError(f"{DATA_PATH} is missing expected columns: {missing}")
        if not existing.empty:
            existing["interval_start_local"] = pd.to_datetime(existing["interval_start_local"], errors="coerce")
            valid = existing.dropna(subset=["location", "interval_start_local"])
            if not valid.empty:
                # Anchor to the earliest "last hour" across zones, so no zone is left with a gap
                last_complete = valid.groupby("location")["interval_start_local"].max().min()
                if pd.notna(last_complete):
                    return last_complete - timedelta(hours=SAFETY_BUFFER_HOURS)

    return now - timedelta(hours=DEFAULT_LOOKBACK_HOURS)


def update_rtm_prices(lookback_hours=None):
    api_key = os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError("GRIDSTATUS_API_KEY environment variable is not set.")

    client = GridStatusClient(api_key=api_key)

    now = pd.Timestamp.now(tz="UTC").tz_convert("-05:00")
    start_dt = determine_start(now, lookback_hours)
    start = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end = now.strftime("%Y-%m-%d %H:%M:%S")  # RTM is real-time, never in the future

    print(f"Fetching {DATASET_ID} from {start} to {end}...")
    try:
        new_df = client.get_dataset(
            dataset=DATASET_ID,
            start=start,
            end=end,
            publish_time="latest",
            timezone="market",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {DATASET_ID} ({start} to {end}): {exc}") from exc

    if new_df is None or new_df.empty:
        raise RuntimeError(f"{DATASET_ID} returned no rows for window {start} to {end}.")

    missing = [c for c in REQUIRED_COLS if c not in new_df.columns]
    if missing:
        raise RuntimeError(f"{DATASET_ID} response is missing expected columns: {missing}")

    new_df["interval_start_local"] = pd.to_datetime(new_df["interval_start_local"])

    # Aggregate the 5-min RTM data to hourly immediately: the dashboard only ever
    # needs the hourly average, and keeping raw 5-min history would blow past
    # GitHub's 100MB file size limit within a few months.
    new_df["interval_start_local"] = new_df["interval_start_local"].dt.floor("h")
    new_df = new_df.groupby(["location", "interval_start_local"])["lmp"].agg(["mean", "count"]).reset_index()

    # The hour still in progress (say 2 of its 12 intervals) would otherwise be stored as if
    # settled: plotted as a normal hour and scored Won/Lost in the Portfolio instead of
    # Pending. Only each zone's latest hour can be partial; the next fetch starts
    # SAFETY_BUFFER_HOURS before the last stored hour, so it comes back once complete.
    latest = new_df.groupby("location")["interval_start_local"].transform("max")
    partial = (new_df["interval_start_local"] == latest) & (new_df["count"] < INTERVALS_PER_HOUR)
    if partial.any():
        print(f"Dropping the in-progress hour ({new_df.loc[partial, 'interval_start_local'].max()}, "
              f"{int(new_df.loc[partial, 'count'].max())} of {INTERVALS_PER_HOUR} intervals)")
    new_df = new_df[~partial].rename(columns={"mean": "lmp"})[REQUIRED_COLS]

    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH)
        existing_missing = [c for c in REQUIRED_COLS if c not in existing_df.columns]
        if existing_missing:
            raise ValueError(f"{DATA_PATH} is missing expected columns: {existing_missing}")
        existing_df["interval_start_local"] = pd.to_datetime(existing_df["interval_start_local"])
        parts = [part for part in (existing_df, new_df) if not part.empty]
        combined = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=REQUIRED_COLS)
    else:
        if new_df.empty:
            raise RuntimeError(f"{DATASET_ID} only returned partial hour data for {start} to {end}; nothing to save.")
        combined = new_df

    combined = combined.drop_duplicates(subset=["interval_start_local", "location"], keep="last")
    combined = combined.sort_values(["location", "interval_start_local"])
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(combined)} rows to {DATA_PATH} (window: {start} to {end})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh IESO RTM prices.")
    parser.add_argument(
        "--hours", type=int, default=None,
        help="Hours to look back from now. If omitted, automatically resumes from the last saved data point."
    )
    args = parser.parse_args()
    update_rtm_prices(lookback_hours=args.hours)
