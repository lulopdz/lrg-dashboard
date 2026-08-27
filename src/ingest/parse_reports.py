import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def parse_reports():
    reports_dir = os.path.join('data', 'reports')
    xml_files = glob.glob(os.path.join(reports_dir, '*.xml'))
    
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
    
    dam_path = os.path.join('data', 'ieso_dam_prices.csv')
    rtm_path = os.path.join('data', 'ieso_rtm_prices.csv')
    
    if not os.path.exists(dam_path) or not os.path.exists(rtm_path):
        print("Price files not found. Cannot calculate PnL.")
        # Just save the ops
        df_ops.to_csv(os.path.join('data', 'historical_pnl.csv'), index=False)
        return

    dam = pd.read_csv(dam_path, parse_dates=['interval_start_local'])
    dam['date'] = dam['interval_start_local'].dt.date
    dam['date'] = pd.to_datetime(dam['date'])
    dam['hour'] = dam['interval_start_local'].dt.hour + 1
    
    rtm = pd.read_csv(rtm_path, parse_dates=['interval_start_local'])
    rtm['date'] = rtm['interval_start_local'].dt.date
    rtm['date'] = pd.to_datetime(rtm['date'])
    rtm['hour'] = rtm['interval_start_local'].dt.hour + 1
    
    dam_ott = dam[dam['location'] == 'OTTAWA'].copy()
    rtm_ott = rtm[rtm['location'] == 'OTTAWA'].copy()
    
    prices = pd.merge(dam_ott[['date', 'hour', 'lmp']], rtm_ott[['date', 'hour', 'lmp']], on=['date', 'hour'], suffixes=('_dam', '_rtm'), how='inner')
    prices['spread'] = prices['lmp_dam'] - prices['lmp_rtm']
    
    df_merged = pd.merge(df_ops, prices, on=['date', 'hour'], how='left')
    
    # PnL = EnergyMW * (DAM - RTM)
    df_merged['pnl'] = df_merged['energy_mw'] * df_merged['spread']
    
    df_merged = df_merged.sort_values(['date', 'hour'])
    
    out_path = os.path.join('data', 'historical_pnl.csv')
    df_merged.to_csv(out_path, index=False)
    print(f"Processed {len(xml_files)} reports. Saved historical PnL to {out_path}")

if __name__ == '__main__':
    parse_reports()
