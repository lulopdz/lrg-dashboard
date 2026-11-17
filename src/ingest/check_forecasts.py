"""Fail loudly when a day's live forecast run is missing.

Between 2026-11-03 and 2026-11-14 the Apps Script trigger that fires daily.yml at 9:00 stopped
firing. The GitHub cron backup landed hours late, after tomorrow's DAM was out, so
run_forecasts.py correctly skipped the pre_dam vintage with a ::notice:: and exit 0 -- and every
run finished green. Twelve days went by without a live pre-DAM forecast before anyone noticed.

This check looks in each zone's spread forecast history (the spread is the last model a run
writes, so its rows mean the whole zone ran) for live rows -- not walk-forward reconstructions --
of the requested vintage and target day, and exits 1 if any zone has none, which turns the
workflow run red and sends GitHub's failure email.

    python src/ingest/check_forecasts.py --vintage pre_dam --day tomorrow   # daily.yml, forecast_pm.yml
    python src/ingest/check_forecasts.py --vintage post_dam --day today     # daily.yml, the morning after

Stdlib only on purpose: the verification jobs that run it do no pip install."""
import argparse
import csv
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MARKET_TZ = timezone(timedelta(hours=-5))  # same fixed EST anchor as io_archive.market_today
# Mirrors src/forecast/config.ZONES (not imported: that module pulls in pandas).
ZONES = ("OTTAWA", "TORONTO", "NIAGARA")
DASHBOARD_ZONE = "OTTAWA"  # files without a zone suffix
VINTAGES = ("pre_dam", "post_dam")


def history_path(zone, data_dir=DATA_DIR):
    suffix = "" if zone == DASHBOARD_ZONE else f"_{zone}"
    return Path(data_dir) / f"spread_forecast_history{suffix}.csv"


def live_hours(path, target_date, vintage):
    """Distinct hours with a live (non-backfilled) row for target_date and vintage."""
    hours = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("target_date", "vintage", "hour") if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")
        for row in reader:
            if row["target_date"] != target_date or row["vintage"] != vintage:
                continue
            if (row.get("backfilled") or "").strip().lower() == "true":
                continue
            hours.add(row["hour"])
    return hours


def check(target_date, vintage, zones=ZONES, data_dir=DATA_DIR):
    """Returns the list of problems (empty = all zones have their live forecast)."""
    problems = []
    for zone in zones:
        path = history_path(zone, data_dir)
        if not path.exists():
            problems.append(f"{zone}: {path.name} does not exist")
            continue
        try:
            hours = live_hours(path, target_date, vintage)
        except (OSError, ValueError) as exc:
            problems.append(f"{zone}: failed to read {path.name}: {exc}")
            continue
        print(f"{zone:<10} {vintage} {target_date}: {len(hours)} live hours")
        if not hours:
            problems.append(f"{zone}: no live {vintage} forecast for {target_date}")
    return problems


def resolve_day(day, today=None):
    today = today or datetime.now(MARKET_TZ).date()
    if day == "today":
        return today.isoformat()
    if day == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    return date.fromisoformat(day).isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Exit 1 if a zone has no live forecast for the day.")
    ap.add_argument("--vintage", choices=VINTAGES, required=True)
    ap.add_argument("--day", default="tomorrow",
                    help="'today', 'tomorrow' (market time, EST) or YYYY-MM-DD; the forecast's target day")
    args = ap.parse_args(argv)
    try:
        target = resolve_day(args.day)
    except ValueError:
        sys.exit(f"--day must be 'today', 'tomorrow' or YYYY-MM-DD, got {args.day!r}")

    problems = check(target, args.vintage)
    if problems:
        for p in problems:
            print(f"::error::{p}")
        sys.exit(f"Missing live {args.vintage} forecasts for {target} in {len(problems)} zone(s)")
    print(f"Live {args.vintage} forecasts for {target} are in for all zones")


if __name__ == "__main__":
    main()
