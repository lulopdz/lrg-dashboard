from pathlib import Path

from update_common import fetch_and_merge

DATASET_ID = "ieso_wind_market_participant_forecast"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ieso_wind_forecast.csv"
PAST_HOURS = 24  # re-fetch a safety window in case IESO issues late corrections
FORECAST_HOURS = 48  # wind forecast is published up to ~2 days ahead
KEEP_COLS = ["interval_start_local", "zone", "generation_forecast"]


def update_wind_forecast():
    fetch_and_merge(
        DATASET_ID, DATA_PATH,
        dedup_subset=["interval_start_local", "zone"], sort_by=["zone", "interval_start_local"],
        past_hours=PAST_HOURS, forecast_hours=FORECAST_HOURS, keep_cols=KEEP_COLS,
    )


if __name__ == "__main__":
    update_wind_forecast()
