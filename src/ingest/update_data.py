import argparse
from pathlib import Path

import pandas as pd

from update_common import fetch_and_merge

DATASET_ID = "ieso_lmp_day_ahead_hourly_virtual_zonal"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ieso_dam_prices.csv"
PAST_HOURS = 48  # re-fetch a safety window in case IESO issues late corrections
FORECAST_HOURS = 36  # DAM is published a day ahead, so the full next day is already available
KEEP_COLS = ["interval_start_local", "location", "lmp"]  # same stored schema as ieso_rtm_prices.csv
PUBLISH_TIME_EST = pd.Timedelta(hours=13, minutes=30)  # IESO posts tomorrow's DAM ~13:30 market time (EST, no DST)


def dam_is_current(now=None):
    """True when the file already holds every DAM day IESO has published so far: today's, plus
    tomorrow's once the ~13:30 EST publication has passed. DAM prices are final once posted, so
    until the next publication a fetch can only return what is already stored -- and each one
    is a request off the 250/month GridStatus quota the workflows share."""
    if not DATA_PATH.exists():
        return False
    if now is None:
        now = pd.Timestamp.now(tz="UTC").tz_convert("-05:00")
    expected_day = now.normalize()
    if now - expected_day >= PUBLISH_TIME_EST:
        expected_day += pd.Timedelta(days=1)
    latest = pd.read_csv(DATA_PATH, parse_dates=["interval_start_local"])["interval_start_local"].max()
    return latest.normalize() >= expected_day


def update_dam_prices(skip_if_current=False):
    if skip_if_current and dam_is_current():
        print("DAM prices already cover every day published so far, skipping fetch.")
        return
    fetch_and_merge(
        DATASET_ID, DATA_PATH,
        dedup_subset=["interval_start_local", "location"], sort_by=["location", "interval_start_local"],
        past_hours=PAST_HOURS, forecast_hours=FORECAST_HOURS, keep_cols=KEEP_COLS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh IESO DAM prices.")
    parser.add_argument(
        "--skip-if-current", action="store_true",
        help="Do nothing when the file already has every DAM day published so far (saves an API request)."
    )
    args = parser.parse_args()
    update_dam_prices(skip_if_current=args.skip_if_current)
