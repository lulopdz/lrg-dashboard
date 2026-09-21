"""Fail loudly when the Real-Time price file is stale.

The daily workflow downloads with continue-on-error so one dead source doesn't block the rest
of the site, but that also let an exhausted GridStatus request quota go unnoticed for a day:
every run finished green while data/ieso_rtm_prices.csv stayed frozen (20-21/09/2026). This
check turns the run red when the freshest RT hour, judged by the zone furthest behind, is
older than --max-hours, so the failure shows up in the Actions list and the notification
email instead of only as a warning annotation nobody opens.

Stdlib only on purpose: the verification job that runs it does no pip install."""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ieso_rtm_prices.csv"


def latest_per_zone(path):
    latest = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["interval_start_local"])
            if row["location"] not in latest or ts > latest[row["location"]]:
                latest[row["location"]] = ts
    return latest


def main():
    parser = argparse.ArgumentParser(description="Exit 1 if the RT price file is stale.")
    parser.add_argument(
        "--max-hours", type=float, default=4,
        help="Oldest acceptable age of the last stored hour, in the zone furthest behind (default 4)."
    )
    args = parser.parse_args()

    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} does not exist")
    latest = latest_per_zone(DATA_PATH)
    if not latest:
        sys.exit(f"{DATA_PATH} has no rows")

    now = datetime.now(timezone.utc)
    for zone, ts in sorted(latest.items()):
        print(f"{zone:<10} last hour {ts}  ({(now - ts).total_seconds() / 3600:.1f} h ago)")

    # Same anchor update_rtm.py resumes from: the zone furthest behind, so a single-zone gap
    # counts as stale too.
    zone, ts = min(latest.items(), key=lambda kv: kv[1])
    age_hours = (now - ts).total_seconds() / 3600
    if age_hours > args.max_hours:
        sys.exit(f"RT prices are stale: {zone} last hour is {ts}, {age_hours:.1f} h ago "
                 f"(limit {args.max_hours:g} h)")
    print(f"RT prices are fresh: oldest zone ({zone}) is {age_hours:.1f} h behind")


if __name__ == "__main__":
    main()
