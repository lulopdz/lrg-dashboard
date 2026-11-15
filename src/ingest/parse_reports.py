"""Portfolio positions and their PnL.

data/portfolio_positions.csv is the source of truth for every participation: one row per
(date, hour, resource) with the MW IESO scheduled. The raw IESO reports it comes from
(DAScheduledEnergy2, marked confidential) stay out of the public repo: data/reports/ is
git-ignored and lives only in the local Drive folder. Dropping a new XML there and running this
script folds its rows into positions; with no XMLs around (GitHub Actions) it just reprices what
positions already holds against the latest DAM/RT and rewrites data/historical_pnl.csv.

--positions-only stops after the positions: what ops/subir_reportes.ps1 runs, since the push it
makes has GitHub Actions reprice anyway, and leaving historical_pnl.csv untouched locally keeps the
next git pull from tripping over it."""
import argparse
import glob
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
POSITIONS_PATH = DATA_DIR / 'portfolio_positions.csv'
POSITION_COLS = ['date', 'hour', 'resource_name', 'resource_type', 'energy_mw']
KEY_COLS = ['date', 'hour', 'resource_name']
PRICE_COLS = ['interval_start_local', 'location', 'lmp']


def zone_of(resource_name):
    """The virtual zonal hub a resource trades at: 'OTTAWA_BID:HUB' -> 'OTTAWA'. Its PnL has to
    use that hub's prices -- every resource so far is an OTTAWA one, but hard-coding OTTAWA
    would silently price a TORONTO trade at Ottawa's spread."""
    if not isinstance(resource_name, str) or '_' not in resource_name.split(':')[0]:
        raise ValueError(f"Unexpected resource_name format: {resource_name!r}")
    return resource_name.split(':')[0].rsplit('_', 1)[0]


def read_xml_reports(xml_files):
    """Every hourly schedule in the given DAScheduledEnergy2 files, as POSITION_COLS rows."""
    rows = []
    failed = []

    # The XML namespace used in the documents
    ns = {'ns': 'http://www.ieso.ca/schema'}

    for file in xml_files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()

            doc_body = root.find('ns:DocBody', ns)
            if doc_body is None:
                continue

            delivery_date = (doc_body.findtext('ns:DeliveryDate', namespaces=ns) or '').strip()

            for resource in doc_body.findall('ns:Resource', ns):
                res_name = (resource.findtext('ns:ResourceName', namespaces=ns) or '').strip()
                res_type = (resource.findtext('ns:ResourceType', namespaces=ns) or '').strip()

                sched_energies = resource.find('ns:ScheduleEnergies', ns)
                if sched_energies is not None:
                    hourly_energies = sched_energies.find('ns:HourlyEnergies', ns)
                    if hourly_energies is not None:
                        for he in hourly_energies.findall('ns:HourlyEnergy', ns):
                            hour_text = (he.findtext('ns:DeliveryHour', namespaces=ns) or '').strip()
                            mw_text = (he.findtext('ns:EnergyMW', namespaces=ns) or '').strip()
                            if not hour_text and not mw_text:
                                continue
                            if not hour_text or not mw_text:
                                raise ValueError("Incomplete HourlyEnergy element (missing DeliveryHour or EnergyMW)")
                            if not delivery_date:
                                raise ValueError("Missing DeliveryDate in DocBody")
                            if not res_name:
                                raise ValueError("Missing ResourceName in Resource")
                            hour = int(hour_text)
                            if not (1 <= hour <= 24):
                                raise ValueError(f"DeliveryHour out of range (1..24): {hour}")
                            rows.append({
                                'date': delivery_date,
                                'hour': hour,
                                'resource_name': res_name,
                                'resource_type': res_type,
                                'energy_mw': float(mw_text)
                            })
        except (ET.ParseError, OSError, ValueError) as e:
            print(f"Error parsing {file}: {e}", file=sys.stderr)
            failed.append((file, e))

    if failed:
        summary = "; ".join(f"{f}: {e}" for f, e in failed)
        raise RuntimeError(f"Failed to parse {len(failed)} XML report(s): {summary}")

    return pd.DataFrame(rows, columns=POSITION_COLS)


def update_positions():
    """portfolio_positions.csv plus whatever XMLs sit in data/reports/, saved back and returned."""
    if POSITIONS_PATH.exists():
        known = pd.read_csv(POSITIONS_PATH, dtype={'date': str})
        missing = [c for c in POSITION_COLS if c not in known.columns]
        if missing:
            raise ValueError(f"{POSITIONS_PATH} is missing expected columns: {missing}")
    else:
        known = pd.DataFrame(columns=POSITION_COLS)

    xml_files = sorted(glob.glob(str(DATA_DIR / 'reports' / '*.xml')))
    from_xml = read_xml_reports(xml_files)
    if xml_files:
        print(f"Read {len(xml_files)} reports from data/reports ({len(from_xml)} hourly rows)")

    # A report always beats what positions already had for the same hour (IESO re-issued it), and
    # among reports a later file name wins -- the same rule as when the XMLs were the only source.
    parts = [part for part in (known, from_xml) if not part.empty]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=POSITION_COLS)
    df = df.drop_duplicates(KEY_COLS, keep='last')
    if not df.empty:
        null_cols = [c for c in KEY_COLS + ['energy_mw'] if df[c].isna().any()]
        if null_cols:
            raise ValueError(f"Positions contain missing values in required column(s): {null_cols}")
    df['hour'] = df['hour'].astype(int)
    df['energy_mw'] = df['energy_mw'].astype(float)
    df = df.sort_values(KEY_COLS).reset_index(drop=True)

    if xml_files:
        added = len(df) - len(known)
        POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(POSITIONS_PATH, index=False, lineterminator='\n')
        print(f"Positions: {len(df)} rows ({added:+d}) saved to {POSITIONS_PATH}")
    return df


def _read_prices(path):
    df = pd.read_csv(path)
    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    df['interval_start_local'] = pd.to_datetime(df['interval_start_local'])
    df['date'] = pd.to_datetime(df['interval_start_local'].dt.date)
    df['hour'] = df['interval_start_local'].dt.hour + 1
    return df


def parse_reports():
    df_ops = update_positions()

    if df_ops.empty:
        print("No operations found in positions or reports.")
        return

    df_ops['date'] = pd.to_datetime(df_ops['date'])
    df_ops['location'] = df_ops['resource_name'].map(zone_of)

    dam_path = DATA_DIR / 'ieso_dam_prices.csv'
    rtm_path = DATA_DIR / 'ieso_rtm_prices.csv'
    out_path = DATA_DIR / 'historical_pnl.csv'

    if not dam_path.exists() or not rtm_path.exists():
        print("Price files not found. Cannot calculate PnL.")
        # Just save the ops
        df_ops.drop(columns='location').to_csv(out_path, index=False)
        return

    dam = _read_prices(dam_path)
    rtm = _read_prices(rtm_path)

    keys = ['location', 'date', 'hour']
    if dam.empty or rtm.empty:
        print("Warning: price file is empty; positions will stay Pending")
        prices = pd.DataFrame(columns=keys + ['lmp_dam', 'lmp_rtm', 'spread'])
    else:
        prices = pd.merge(dam[keys + ['lmp']], rtm[keys + ['lmp']], on=keys, suffixes=('_dam', '_rtm'), how='inner')
        prices['spread'] = prices['lmp_dam'] - prices['lmp_rtm']

        unknown = sorted(set(df_ops['location']) - set(prices['location']))
        if unknown:
            print(f"Warning: no prices for zone(s) {unknown} -- their hours stay Pending")

    df_merged = pd.merge(df_ops, prices, on=keys, how='left').drop(columns='location')

    # PnL = EnergyMW * (DAM - RTM)
    df_merged['pnl'] = df_merged['energy_mw'] * df_merged['spread']

    df_merged = df_merged.sort_values(KEY_COLS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_path, index=False)
    print(f"Priced {len(df_ops)} position rows. Saved historical PnL to {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Fold IESO participation reports into positions and price them.")
    ap.add_argument('--positions-only', action='store_true',
                    help="update data/portfolio_positions.csv and stop (no historical_pnl.csv)")
    try:
        if ap.parse_args().positions_only:
            update_positions()
        else:
            parse_reports()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
