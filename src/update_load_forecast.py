from pathlib import Path

from update_common import fetch_and_merge

DATASET_ID = "ieso_zonal_load_forecast_hourly"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ieso_load_forecast.csv"
PAST_HOURS = 24  # re-fetch a safety window in case IESO issues late corrections
FORECAST_HOURS = 48  # zonal load forecast is published up to ~2 days ahead


def update_load_forecast():
    fetch_and_merge(
        DATASET_ID, DATA_PATH,
        dedup_subset=["interval_start_local"], sort_by="interval_start_local",
        past_hours=PAST_HOURS, forecast_hours=FORECAST_HOURS,
    )


if __name__ == "__main__":
    update_load_forecast()
