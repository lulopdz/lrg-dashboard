import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from gridstatusio import GridStatusClient

DATASET_ID = "ieso_lmp_real_time_5_min_virtual_zonal"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ieso_rtm_prices.csv"
PAST_HOURS = 24  # re-fetch a safety window in case IESO issues late corrections


def update_rtm_prices():
    api_key = os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError("GRIDSTATUS_API_KEY environment variable is not set.")

    client = GridStatusClient(api_key=api_key)

    now = pd.Timestamp.now(tz="UTC").tz_convert("-05:00")
    start = (now - timedelta(hours=PAST_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
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
    print(f"Saved {len(combined)} rows to {DATA_PATH}")


if __name__ == "__main__":
    update_rtm_prices()
