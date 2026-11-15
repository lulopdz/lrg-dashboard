"""DAM and RT zonal prices straight from IESO's public reports, no API key and no quota.

Same numbers as the GridStatus datasets the daily workflow uses (checked value-for-value on
2026-09-22: 216/216 identical), same CSV schema, same merge rule (a re-fetched hour
overwrites the stored one). Exists so the afternoon post-DAM forecast run can pull tomorrow's
DAM and today's RT without spending any of the 250 GridStatus requests a month.

    python src/ingest/update_prices_ieso.py dam            # tomorrow's DAM (IESO posts it ~12:35 EST)
    python src/ingest/update_prices_ieso.py dam --date 2026-09-22
    python src/ingest/update_prices_ieso.py rtm            # today's RT so far, hourly means of the 5-min zonal prices
    python src/ingest/update_prices_ieso.py rtm --date 2026-09-21 --hours 1-14

Exit code 3 when the report isn't published yet, so a workflow can tell "too early" from a
real failure. Market time is EST all year (IESO doesn't observe DST), hence the fixed -05:00."""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

BASE = "https://reports-public.ieso.ca/public"
NS = {"i": "http://www.ieso.ca/schema"}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DAM_PATH = DATA_DIR / "ieso_dam_prices.csv"
RTM_PATH = DATA_DIR / "ieso_rtm_prices.csv"
NOT_PUBLISHED = 3


def market_now():
    return pd.Timestamp.now(tz="UTC").tz_convert("-05:00")


def fetch(url):
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"HTTP request failed for {url}: {exc}") from exc
    try:
        return ET.fromstring(r.content)
    except ET.ParseError as exc:
        raise RuntimeError(f"Invalid XML from {url}: {exc}") from exc


def dam_rows(date):
    """Hourly 'Zonal Price' for the nine virtual zonal hubs of one delivery date."""
    root = fetch(f"{BASE}/DAHourlyZonal/PUB_DAHourlyZonal_{date:%Y%m%d}.xml")
    if root is None:
        return None
    rows = []
    for zone in root.iter(f"{{{NS['i']}}}TransactionZone"):
        zone_text = (zone.findtext("i:ZoneName", namespaces=NS) or "").strip()
        if not zone_text:
            raise ValueError(f"DAM XML for {date} has TransactionZone without ZoneName")
        name = zone_text.replace(":HUB", "")
        for comp in zone.findall("i:Components", NS):
            if comp.findtext("i:PriceComponent", namespaces=NS) != "Zonal Price":
                continue
            for dh in comp.findall("i:DeliveryHour", NS):
                hour_text = (dh.findtext("i:Hour", namespaces=NS) or "").strip()
                lmp_text = (dh.findtext("i:LMP", namespaces=NS) or "").strip()
                if not hour_text or not lmp_text:
                    raise ValueError(f"DAM XML for {date} ({name}) has DeliveryHour missing Hour or LMP")
                hour = int(hour_text)
                start = pd.Timestamp(date, tz="-05:00") + pd.Timedelta(hours=hour - 1)
                rows.append((start, name, float(lmp_text)))
    if not rows:
        raise RuntimeError(f"DAM XML for {date} contained no Zonal Price rows")
    return pd.DataFrame(rows, columns=["interval_start_local", "location", "lmp"])


def rtm_rows(date, hours):
    """Hourly mean of the twelve 5-minute zonal prices, per hub, for the given hours-ending.
    Stops at the first hour that isn't published yet."""
    rows = []
    for he in hours:
        root = fetch(f"{BASE}/RealtimeZonalEnergyPrices/PUB_RealtimeZonalEnergyPrices_{date:%Y%m%d}{he:02d}.xml")
        if root is None:
            break
        start = pd.Timestamp(date, tz="-05:00") + pd.Timedelta(hours=he - 1)
        for zone in root.iter(f"{{{NS['i']}}}TransactionZone"):
            zone_text = (zone.findtext("i:ZoneName", namespaces=NS) or "").strip()
            if not zone_text:
                raise ValueError(f"RTM XML for {date} HE {he} has TransactionZone without ZoneName")
            name = zone_text.replace(":HUB", "")
            prices = []
            for ip in zone.findall("i:IntervalPrice", NS):
                price_text = (ip.findtext("i:ZonalPrice", namespaces=NS) or "").strip()
                if not price_text:
                    raise ValueError(f"RTM XML for {date} HE {he} ({name}) has IntervalPrice missing ZonalPrice")
                prices.append(float(price_text))
            if prices:
                rows.append((start, name, sum(prices) / len(prices)))
    return pd.DataFrame(rows, columns=["interval_start_local", "location", "lmp"])


def merge(path, new, sort_by):
    required = ["interval_start_local", "location", "lmp"]
    if path.exists():
        existing = pd.read_csv(path)
        missing = [c for c in required if c not in existing.columns]
        if missing:
            raise ValueError(f"{path} is missing expected columns: {missing}")
        existing["interval_start_local"] = pd.to_datetime(existing["interval_start_local"])
        parts = [part for part in (existing, new) if not part.empty]
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=required)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=["interval_start_local", "location"], keep="last")
    combined = combined.sort_values(sort_by)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    print(f"Saved {len(combined)} rows to {path} (+{len(new)} fetched)")


def main():
    ap = argparse.ArgumentParser(description="IESO public DAM/RT zonal prices into the data CSVs.")
    ap.add_argument("market", choices=["dam", "rtm"])
    ap.add_argument("--date", default=None, help="delivery date YYYY-MM-DD (default: tomorrow for dam, today for rtm; market time)")
    ap.add_argument("--hours", default=None, help="rtm only: hours-ending range like 1-14 (default: 1 through the current hour)")
    args = ap.parse_args()
    now = market_now()
    if args.market == "dam":
        date = pd.Timestamp(args.date).date() if args.date else (now + pd.Timedelta(days=1)).date()
        new = dam_rows(date)
        if new is None:
            print(f"DAM for {date} not published yet")
            sys.exit(NOT_PUBLISHED)
        merge(DAM_PATH, new, ["location", "interval_start_local"])
    else:
        date = pd.Timestamp(args.date).date() if args.date else now.date()
        if args.hours:
            parts = args.hours.split("-")
            if len(parts) != 2:
                ap.error("--hours must be a range like 1-14")
            try:
                lo, hi = int(parts[0]), int(parts[1])
            except ValueError:
                ap.error("--hours must contain integers like 1-14")
            if not (1 <= lo <= hi <= 24):
                ap.error("--hours range must satisfy 1 <= start <= end <= 24")
        else:
            lo, hi = 1, (now.hour if date == now.date() else 24)  # the current hour is still open
        new = rtm_rows(date, range(lo, hi + 1))
        if new.empty:
            print(f"No RT hours published yet for {date}")
            sys.exit(NOT_PUBLISHED)
        merge(RTM_PATH, new, ["location", "interval_start_local"])


if __name__ == "__main__":
    main()
