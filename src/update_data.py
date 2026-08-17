from pathlib import Path

from update_common import fetch_and_merge

DATASET_ID = "ieso_lmp_day_ahead_hourly_virtual_zonal"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ieso_dam_prices.csv"
PAST_HOURS = 48  # re-fetch a safety window in case IESO issues late corrections
FORECAST_HOURS = 36  # DAM is published a day ahead, so the full next day is already available


def update_dam_prices():
    fetch_and_merge(
        DATASET_ID, DATA_PATH,
        dedup_subset=["interval_start_local", "location"], sort_by=["location", "interval_start_local"],
        past_hours=PAST_HOURS, forecast_hours=FORECAST_HOURS,
    )


if __name__ == "__main__":
    update_dam_prices()
