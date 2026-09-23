import glob
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


def zone_of(resource_name):
    """The virtual zonal hub a resource trades at: 'OTTAWA_BID:HUB' -> 'OTTAWA'. Its PnL has to
    use that hub's prices -- every resource so far is an OTTAWA one, but hard-coding OTTAWA
    would silently price a TORONTO trade at Ottawa's spread."""
    return resource_name.split(':')[0].rsplit('_', 1)[0]


def parse_reports():
    xml_files = sorted(glob.glob(str(DATA_DIR / 'reports' / '*.xml')))

    rows = []

    # The XML namespace used in the documents
    ns = {'ns': 'http://www.ieso.ca/schema'}

    for file in xml_files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()

            doc_body = root.find('ns:DocBody', ns)
            if doc_body is None:
                continue

            delivery_date = doc_body.findtext('ns:DeliveryDate', namespaces=ns)

            for resource in doc_body.findall('ns:Resource', ns):
                res_name = resource.findtext('ns:ResourceName', namespaces=ns)
                res_type = resource.findtext('ns:ResourceType', namespaces=ns)

                sched_energies = resource.find('ns:ScheduleEnergies', ns)
                if sched_energies is not None:
                    hourly_energies = sched_energies.find('ns:HourlyEnergies', ns)
                    if hourly_energies is not None:
                        for he in hourly_energies.findall('ns:HourlyEnergy', ns):
                            hour_text = he.findtext('ns:DeliveryHour', namespaces=ns)
                            mw_text = he.findtext('ns:EnergyMW', namespaces=ns)
                            if hour_text and mw_text:
                                hour = int(hour_text)
                                mw = float(mw_text)

                                rows.append({
                                    'date': delivery_date,
                                    'hour': hour,
                                    'resource_name': res_name,
                                    'resource_type': res_type,
                                    'energy_mw': mw
                                })
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    if not rows:
        print("No operations found in reports.")
        return

    df_ops = pd.DataFrame(rows)
    df_ops['date'] = pd.to_datetime(df_ops['date'])
    # A report IESO re-issues for the same day would otherwise count every hour twice; files are
    # read in name order, so the later one wins.
    df_ops = df_ops.drop_duplicates(['date', 'hour', 'resource_name'], keep='last')
    df_ops['location'] = df_ops['resource_name'].map(zone_of)

    dam_path = DATA_DIR / 'ieso_dam_prices.csv'
    rtm_path = DATA_DIR / 'ieso_rtm_prices.csv'
    out_path = DATA_DIR / 'historical_pnl.csv'

    if not dam_path.exists() or not rtm_path.exists():
        print("Price files not found. Cannot calculate PnL.")
        # Just save the ops
        df_ops.drop(columns='location').to_csv(out_path, index=False)
        return

    dam = pd.read_csv(dam_path, parse_dates=['interval_start_local'])
    dam['date'] = dam['interval_start_local'].dt.date
    dam['date'] = pd.to_datetime(dam['date'])
    dam['hour'] = dam['interval_start_local'].dt.hour + 1

    rtm = pd.read_csv(rtm_path, parse_dates=['interval_start_local'])
    rtm['date'] = rtm['interval_start_local'].dt.date
    rtm['date'] = pd.to_datetime(rtm['date'])
    rtm['hour'] = rtm['interval_start_local'].dt.hour + 1

    keys = ['location', 'date', 'hour']
    prices = pd.merge(dam[keys + ['lmp']], rtm[keys + ['lmp']], on=keys, suffixes=('_dam', '_rtm'), how='inner')
    prices['spread'] = prices['lmp_dam'] - prices['lmp_rtm']

    unknown = sorted(set(df_ops['location']) - set(prices['location']))
    if unknown:
        print(f"Warning: no prices for zone(s) {unknown} -- their hours stay Pending")

    df_merged = pd.merge(df_ops, prices, on=keys, how='left').drop(columns='location')

    # PnL = EnergyMW * (DAM - RTM)
    df_merged['pnl'] = df_merged['energy_mw'] * df_merged['spread']

    df_merged = df_merged.sort_values(['date', 'hour'])

    df_merged.to_csv(out_path, index=False)
    print(f"Processed {len(xml_files)} reports. Saved historical PnL to {out_path}")

if __name__ == '__main__':
    parse_reports()
