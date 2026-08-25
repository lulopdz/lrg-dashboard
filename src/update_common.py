"""Shared fetch-and-merge plumbing for the GridStatus-backed update_*.py scripts
(update_data.py, update_load_forecast.py, update_wind_forecast.py): each just calls the
GridStatus API for its own dataset over a rolling past/forecast window, appends the result
to its existing CSV, and drops duplicates keeping the freshest value -- so a late correction
or a firmer forecast always overwrites the earlier guess. update_rtm.py stays separate: it
also aggregates raw 5-min data to hourly and resumes from the last saved point instead of a
fixed lookback, which doesn't fit this shape."""
import os
from datetime import timedelta

import pandas as pd
from gridstatusio import GridStatusClient


def fetch_and_merge(dataset_id, data_path, dedup_subset, sort_by, past_hours, forecast_hours,
                    keep_cols):
    """keep_cols is the exact stored schema: GridStatus returns a wider frame than anything
    here reads (UTC mirrors of the local timestamps, publish/last-modified stamps, DAM's
    energy/congestion/loss breakdown), and keeping those tripled the on-disk size of files the
    daily workflow re-commits. Applied to both sides of the merge, so a file written before
    this existed gets trimmed on its next update instead of needing a separate migration."""
    api_key = os.getenv("GRIDSTATUS_API_KEY")
    if not api_key:
        raise RuntimeError("GRIDSTATUS_API_KEY environment variable is not set.")

    client = GridStatusClient(api_key=api_key)

    now = pd.Timestamp.now(tz="UTC").tz_convert("-05:00")
    start = (now - timedelta(hours=past_hours)).strftime("%Y-%m-%d %H:%M:%S")
    end = (now + timedelta(hours=forecast_hours)).strftime("%Y-%m-%d %H:%M:%S")

    print(f"Fetching {dataset_id} from {start} to {end}...")
    new_df = client.get_dataset(
        dataset=dataset_id,
        start=start,
        end=end,
        publish_time="latest",
        timezone="market",
    )
    new_df["interval_start_local"] = pd.to_datetime(new_df["interval_start_local"])

    missing = [c for c in keep_cols if c not in new_df.columns]
    if missing:
        raise RuntimeError(f"{dataset_id} response is missing expected columns: {missing}")
    new_df = new_df[keep_cols]

    if data_path.exists():
        existing_df = pd.read_csv(data_path, parse_dates=["interval_start_local"])
        combined = pd.concat([existing_df[keep_cols], new_df], axis=0)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
    combined = combined.sort_values(sort_by)
    combined.to_csv(data_path, index=False)
    print(f"Saved {len(combined)} rows to {data_path}")
    return combined
