from pathlib import Path

from update_common import fetch_and_merge

DATASET_ID = "ieso_adequacy_report_forecast"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ieso_adequacy.csv"
PAST_HOURS = 48       # re-fetch a safety window in case IESO reissues a report
FORECAST_HOURS = 48   # IESO publishes up to ~34 days out; we only plot/model two days

# All 103 columns are stored, unlike the other feeds which are trimmed to what the pages read.
# The value here is the generation-mix breakdown itself -- capacity, outages, offered and
# scheduled for every fuel, plus interchange per tie and embedded behind-the-meter wind/solar
# -- so picking a subset now would foreclose exactly the comparisons this dataset is for.
#
# It stays small anyway because of publish_time="latest" in fetch_and_merge: IESO reissues this
# report ~106 times per delivery hour (up to 34 days ahead), and an export that keeps every
# vintage runs 444 MB for the same period this stores in about 5 MB. publish_time_local is kept
# so each row still says which run it came from.
KEEP_COLS = None


def update_adequacy():
    fetch_and_merge(
        DATASET_ID, DATA_PATH,
        dedup_subset=["interval_start_local"], sort_by="interval_start_local",
        past_hours=PAST_HOURS, forecast_hours=FORECAST_HOURS, keep_cols=KEEP_COLS,
    )


if __name__ == "__main__":
    update_adequacy()
