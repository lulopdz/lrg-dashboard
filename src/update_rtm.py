import argparse
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from gridstatusio import GridStatusClient

DATASET_ID = "ieso_lmp_real_time_5_min_virtual_zonal"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ieso_rtm_prices.csv"
DEFAULT_LOOKBACK_HOURS = 24  # used only when there's no existing file to anchor from
SAFETY_BUFFER_HOURS = 2  # re-fetch a small overlap in case IESO revises recent values


def determine_start(now, lookback_hours):
    if lookback_hours is not None:
        return now - timedelta(hours=lookback_hours)

    if DATA_PATH.exists():
        existing = pd.read_csv(DATA_PATH, parse_dates=["interval_start_local"])
        if not existing.empty:
            # Anchor to the earliest "last hour" across zones, so no zone is left with a gap
            last_complete = existing.groupby("location")["interval_start_local"].max().min()
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
    new_df = client.get_dataset(
        dataset=DATASET_ID,
        start=start,
        end=end,
        publish_time="latest",
        timezone="market",
    )
    new_df["interval_start_local"] = pd.to_datetime(new_df["interval_start_local"])

    # Aggregate the 5-min RTM data to hourly immediately: the dashboard only ever
    # needs the hourly average, and keeping raw 5-min history would blow past
    # GitHub's 100MB file size limit within a few months.
    new_df["interval_start_local"] = new_df["interval_start_local"].dt.floor("h")
    new_df = new_df.groupby(["location", "interval_start_local"])["lmp"].mean().reset_index()

    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH, parse_dates=["interval_start_local"])
        combined = pd.concat([existing_df, new_df], axis=0)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["interval_start_local", "location"], keep="last")
    combined = combined.sort_values(["location", "interval_start_local"])
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
