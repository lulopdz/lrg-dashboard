import json
import os

import pandas as pd

from dashboard_data import (
    DAY_OPTION_STRS, DEFAULT_ZONE, GITHUB_OWNER, GITHUB_REPO, LOAD_VARS, SELECTABLE_DATE_STRS,
    TABLE_DAYS, WEATHER_VARS, dam, default_date_idx, default_load_idx, default_weather_date_idx,
    default_weather_idx, default_wind_idx, latest_ts, load_forecast, load_latest_ts,
    load_var_keys, rtm, rtm_latest_ts, spread, table_start_date, today_date, weather,
    weather_latest_ts, weather_var_keys, wind_forecast, wind_latest_ts, wind_zones, zones,
)
from dashboard_figures import (
    build_forecast_fig, build_hourly_fig, build_spread_detail_fig, build_table_fig,
    build_wide_hourly_fig, build_wide_table_fig,
)


def date_options_html(selected_date_str):
    return '\n'.join(
        f'<option value="{d}"{" selected" if d == selected_date_str else ""}>{d}</option>'
        for d in DAY_OPTION_STRS
    )


def options_control(div_id, label_text, option_labels, selected_label):
    """Generic '<select>' control wired into the registerFig/applyFigSelection JS layer.
    Used for zone selectors (DAM/RTM/Spread/Wind) and variable selectors (Weather/Load),
    which differ only in label text and the list of option strings."""
    options = '\n'.join(
        f'<option value="{o}"{" selected" if o == selected_label else ""}>{o}</option>'
        for o in option_labels
    )
    return f"""<div class="controls">
  <label>{label_text}:</label>
  <select id="{div_id}-zone" onchange="applyFigSelection('{div_id}')">
    {options}
  </select>
</div>"""


def weather_date_control(div_id):
    """Independent Day selector for Weather, defaulting to tomorrow. Kept separate from the
    shared global-date bar since DAM/RTM/Spread default to today (RTM has no future data)."""
    return f"""<div class="controls">
  <label>Day:</label>
  <select id="{div_id}-date" onchange="applyFigSelection('{div_id}')">
    {date_options_html(DAY_OPTION_STRS[default_weather_date_idx])}
  </select>
</div>"""


dam_hourly_fig = build_hourly_fig(dam, 'DAM')
dam_table_fig = build_table_fig(dam, 'DAM', palette='Blues')

rtm_hourly_fig = build_hourly_fig(rtm, 'RTM')
rtm_table_fig = build_table_fig(rtm, 'RTM', palette='Oranges')

spread_hourly_fig = build_spread_detail_fig()
spread_table_fig = build_table_fig(spread, 'Spread (DAM - RTM)', diverging=True)

weather_hourly_fig = build_wide_hourly_fig(weather, 'timestamp', WEATHER_VARS, default_weather_idx, 'Weather',
                                            default_day_idx=default_weather_date_idx)
weather_table_fig = build_wide_table_fig(weather, 'timestamp', WEATHER_VARS, default_weather_idx, 'Weather')

load_hourly_fig = build_wide_hourly_fig(load_forecast, 'interval_start_local', LOAD_VARS, default_load_idx, 'Load Forecast')
load_table_fig = build_wide_table_fig(load_forecast, 'interval_start_local', LOAD_VARS, default_load_idx, 'Load Forecast',
                                       colorscale='Viridis')

wind_hourly_fig = build_hourly_fig(wind_forecast, 'Wind Forecast', location_col='zone', value_col='generation_forecast',
                                    y_axis_title='Generation (MW)', zones_list=wind_zones, default_zone_idx=default_wind_idx)
wind_table_fig = build_table_fig(wind_forecast, 'Wind Forecast', palette='Greens', location_col='zone',
                                  value_col='generation_forecast', zones_list=wind_zones, default_zone_idx=default_wind_idx,
                                  colorbar_title='MW', hover_label='Generation', hover_prefix='', hover_suffix=' MW')

def build_forecast_tab(csv_path, meta_path, tab_id, series_label, script_name):
    """A forecast tab (predicted curve + backtest stats + similar-day comparison table),
    shared by the DAM/RTM/Spread Forecast tabs -- only shown once the matching
    predict_*.py script has been run manually. Returns (tab_button_html, tab_content_html)."""
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return '', ''

    forecast = pd.read_csv(csv_path)
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    fig = build_forecast_fig(forecast, meta, series_label=series_label)

    backtest = meta.get('backtest') or {}
    model_mae = f"${backtest['model_mae']:.2f}" if backtest else 'n/a'
    naive_mae = f"${backtest['naive_mae']:.2f}" if backtest else 'n/a'
    backtest_days_label = round(backtest['n_test_hours'] / 24) if backtest else 'n/a'
    analog_label = meta.get('analog_date') or 'n/a'

    recommended = meta.get('recommended_hour') or {}
    confident_hour_label = (
        f"Hour {recommended['hour']} (±${recommended['expected_error']:.2f})" if recommended.get('hour') else 'n/a'
    )

    def pct_diff_label(target, analog):
        if analog == 0:
            return 'n/a'
        return f"{(target - analog) / analog * 100:+.1f}%"

    def row_html(c, c2=None):
        cells = [c['label'], f"{c['weight']:g}x", f"{c['target']:.1f} {c['unit']}",
                 f"{c['analog']:.1f} {c['unit']}", pct_diff_label(c['target'], c['analog'])]
        if c2:
            cells += [f"{c2['analog']:.1f} {c2['unit']}", pct_diff_label(c2['target'], c2['analog'])]
        return "<tr>" + "".join(f"<td>{v}</td>" for v in cells) + "</tr>"

    comparison_rows = meta.get('analog_comparison') or []
    comparison_rows_2 = meta.get('analog_comparison_2') or []
    analog_label_2 = meta.get('analog_date_2')

    if comparison_rows_2:
        comparison_rows_html = '\n'.join(row_html(c, c2) for c, c2 in zip(comparison_rows, comparison_rows_2))
        first_header = f"#1: {analog_label} (actual)"
        extra_header = f"<th>#2: {analog_label_2} (actual)</th><th>% diff</th>"
    else:
        comparison_rows_html = '\n'.join(row_html(c) for c in comparison_rows)
        first_header = f"{analog_label} (actual)"
        extra_header = ""

    comparison_table_html = f"""
<h3>Why this day? Tomorrow's forecast vs. the closest historical day{'s' if comparison_rows_2 else ''}</h3>
<table class="compare-table">
  <thead><tr><th>Variable</th><th>Weight</th><th>Tomorrow (forecast)</th><th>{first_header}</th><th>% diff</th>{extra_header}</tr></thead>
  <tbody>
{comparison_rows_html}
  </tbody>
</table>""" if comparison_rows else ''

    tab_content_html = f"""
<div id="tab-{tab_id}" class="tab-content">
<div>
  <a class="refresh-btn" href="https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/predict.yml" target="_blank" rel="noopener">
    Refresh Forecasts (opens GitHub Actions)
  </a>
</div>
<h2>{series_label} Forecast - {meta.get('zone')} ({meta.get('target_date')})</h2>
<p style="color:#aaa; max-width:800px;">Tomorrow's {series_label} hasn't happened yet -- this is a model
prediction, not historical data. Generated by <code>src/{script_name}</code>, run manually via the button above
(not part of the daily automated data refresh).</p>
<div class="stat-row">
  <div class="stat-tile"><div class="stat-label">Target date</div><div class="stat-value">{meta.get('target_date')}</div></div>
  <div class="stat-tile"><div class="stat-label">Model MAE ({backtest_days_label}d backtest)</div><div class="stat-value">{model_mae}</div></div>
  <div class="stat-tile"><div class="stat-label">Naive baseline MAE</div><div class="stat-value">{naive_mae}</div></div>
  <div class="stat-tile"><div class="stat-label">Most similar day</div><div class="stat-value">{analog_label}</div></div>
  <div class="stat-tile"><div class="stat-label">Most confident hour</div><div class="stat-value">{confident_hour_label}</div></div>
</div>
{fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'{tab_id}-forecast')}
{comparison_table_html}
</div>
"""
    tab_button_html = f'<button class="tab-btn" onclick="showTab(\'{tab_id}\', this)">{series_label} Forecast</button>'
    return tab_button_html, tab_content_html


dam_forecast_tab_button, dam_forecast_tab_html = build_forecast_tab(
    'data/dam_forecast.csv', 'data/dam_forecast_meta.json', 'forecast', 'DAM', 'predict_dam.py')
rtm_forecast_tab_button, rtm_forecast_tab_html = build_forecast_tab(
    'data/rtm_forecast.csv', 'data/rtm_forecast_meta.json', 'rtm-forecast', 'RTM', 'predict_rtm.py')
spread_forecast_tab_button, spread_forecast_tab_html = build_forecast_tab(
    'data/spread_forecast.csv', 'data/spread_forecast_meta.json', 'spread-forecast', 'Spread', 'predict_spread.py')


def _pivot_to_js(df, time_col, location_col, value_col, group_keys):
    """Shared pivot logic: returns a dict keyed by group_key → {dates, hours, values}."""
    df_t = df[(df[time_col].dt.date >= table_start_date) &
              (df[time_col].dt.date <= today_date)].copy()
    df_t['date'] = df_t[time_col].dt.date.astype(str)
    data = {}
    for key in group_keys:
        pivot = (df_t[df_t[location_col] == key]
                 .pivot_table(index='date', columns='hour', values=value_col, aggfunc='mean')
                 .reindex(index=SELECTABLE_DATE_STRS, columns=range(1, 25)))
        data[key] = {
            'dates': SELECTABLE_DATE_STRS,
            'hours': list(range(1, 25)),
            'values': [[None if pd.isna(v) else round(float(v), 2) for v in row]
                       for row in pivot.values]
        }
    return data


def table_data_js(df, var_name, zones_list=None, location_col='location', value_col='lmp'):
    zones_list = zones_list if zones_list is not None else zones
    data = _pivot_to_js(df, 'interval_start_local', location_col, value_col, zones_list)
    return f"const {var_name} = {json.dumps(data)};"


def wide_table_data_js(df, time_col, var_map, var_name):
    df_t = df[(df[time_col].dt.date >= table_start_date) &
              (df[time_col].dt.date <= today_date)].copy()
    df_t['date'] = df_t[time_col].dt.date.astype(str)
    data = {}
    for var, (label, _) in var_map.items():
        pivot = (df_t.pivot_table(index='date', columns='hour', values=var, aggfunc='mean')
                 .reindex(index=SELECTABLE_DATE_STRS, columns=range(1, 25)))
        data[label] = {
            'dates': SELECTABLE_DATE_STRS,
            'hours': list(range(1, 25)),
            'values': [[None if pd.isna(v) else round(float(v), 2) for v in row]
                       for row in pivot.values]
        }
    return f"const {var_name} = {json.dumps(data)};"


dam_table_data_js     = table_data_js(dam, 'DAM_TABLE_DATA')
rtm_table_data_js     = table_data_js(rtm, 'RTM_TABLE_DATA')
spread_table_data_js  = table_data_js(spread, 'SPREAD_TABLE_DATA')
weather_table_data_js = wide_table_data_js(weather, 'timestamp', WEATHER_VARS, 'WEATHER_TABLE_DATA')
load_table_data_js    = wide_table_data_js(load_forecast, 'interval_start_local', LOAD_VARS, 'LOAD_TABLE_DATA')
wind_table_data_js    = table_data_js(wind_forecast, 'WIND_TABLE_DATA', zones_list=wind_zones,
                                       location_col='zone', value_col='generation_forecast')

# 4. Assemble the HTML page with tabs (DAM / RTM / Spread / Weather / Load Forecast / Wind Forecast)
# and shared zone/day controls
os.makedirs('docs', exist_ok=True)

ZONES_JSON = json.dumps(zones)
DATES_JSON = json.dumps(DAY_OPTION_STRS)
WEATHER_LABELS_JSON = json.dumps([WEATHER_VARS[v][0] for v in weather_var_keys])
WEATHER_Y_AXIS_TITLES = [f'{WEATHER_VARS[v][0]} ({WEATHER_VARS[v][1]})' for v in weather_var_keys]

LOAD_LABELS_JSON = json.dumps([LOAD_VARS[v][0] for v in load_var_keys])
LOAD_Y_AXIS_TITLES = [f'{LOAD_VARS[v][0]} ({LOAD_VARS[v][1]})' for v in load_var_keys]

WIND_ZONES_JSON = json.dumps(wind_zones)


def zone_control(div_id):
    return options_control(div_id, 'Zone', zones, DEFAULT_ZONE)


def weather_control(div_id):
    return options_control(div_id, 'Variable', [WEATHER_VARS[v][0] for v in weather_var_keys],
                            WEATHER_VARS[weather_var_keys[default_weather_idx]][0])


def load_control(div_id):
    return options_control(div_id, 'Zone', [LOAD_VARS[v][0] for v in load_var_keys],
                            LOAD_VARS[load_var_keys[default_load_idx]][0])


def wind_control(div_id):
    return options_control(div_id, 'Zone', wind_zones, wind_zones[default_wind_idx])


def register_fig(div_id, traces_per_combo, title_prefix, zones_json=None, y_axis_titles=None, date_sel_id=None):
    zones_json = zones_json if zones_json is not None else ZONES_JSON
    y_axis_json = json.dumps(y_axis_titles) if y_axis_titles else 'null'
    date_sel_json = json.dumps(date_sel_id) if date_sel_id else 'null'
    return (f"<script>registerFig('{div_id}', {zones_json}, {DATES_JSON}, {traces_per_combo}, "
            f"'{title_prefix}', {y_axis_json}, {date_sel_json});</script>")


html = f"""<html>
<head>
<meta charset="utf-8">
<title>DAM Dashboard</title>
<style>
  body {{ background:#111; color:#eee; font-family:Arial, sans-serif; margin:0; padding:24px; }}
  h2 {{ color:#ddd; border-bottom:1px solid #333; padding-bottom:6px; }}
  footer {{ color:#888; font-size:12px; margin-top:20px; }}
  .tabs {{ display:flex; gap:8px; margin-bottom:0; }}
  .tab-btn {{
    background:#1e1e1e; color:#ccc; border:1px solid #333; border-radius:6px 6px 0 0;
    padding:10px 20px; cursor:pointer; font-size:14px;
  }}
  .tab-btn.active {{ background:#2c2c2c; color:#fff; border-bottom:2px solid #3498db; }}
  .refresh-btn {{
    display:inline-block; background:#3498db; color:#fff; border:none; border-radius:4px;
    padding:8px 16px; cursor:pointer; font-size:13px; margin-bottom:8px; text-decoration:none;
  }}
  .refresh-btn:hover {{ background:#2980b9; }}
  .global-day-bar {{
    background:#1a1a1a; border:1px solid #333; border-radius:4px;
    padding:10px 16px; margin:16px 0; display:flex; align-items:center; gap:10px;
  }}
  .controls {{ margin:8px 0; }}
  select {{ background:#1e1e1e; color:#eee; border:1px solid #444; border-radius:4px; padding:4px 8px; }}
  /* max-height:0 (instead of display:none) keeps the container's width intact so
     Plotly's auto-sizing doesn't collapse hidden charts to zero width on first render */
  .tab-content {{ max-height:0; overflow:hidden; }}
  .tab-content.active {{ max-height:none; }}
  .copy-btn {{
    background:#2c2c2c; color:#aaa; border:1px solid #444; border-radius:4px;
    padding:3px 10px; cursor:pointer; font-size:12px; margin-left:12px; vertical-align:middle;
  }}
  .copy-btn:hover {{ background:#444; color:#fff; }}
  .stat-row {{ display:flex; flex-wrap:wrap; gap:16px; margin:16px 0; }}
  .stat-tile {{ background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:12px 20px; }}
  .stat-label {{ color:#888; font-size:12px; }}
  .stat-value {{ color:#eee; font-size:22px; font-weight:600; margin-top:4px; }}
  .compare-table {{ border-collapse:collapse; margin:12px 0 24px; }}
  .compare-table th, .compare-table td {{ padding:6px 20px 6px 0; text-align:right; border-bottom:1px solid #333; }}
  .compare-table th:first-child, .compare-table td:first-child {{ text-align:left; }}
  .compare-table th {{ color:#888; font-weight:500; font-size:13px; }}
  .compare-table td {{ color:#eee; font-variant-numeric:tabular-nums; }}
  .compare-table td:first-child {{ color:#ccc; }}
</style>
</head>
<body>

<script>
function showTab(name, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}

const FIG_CONFIGS = {{}};

function registerFig(divId, zonesList, datesList, tracesPerCombo, titlePrefix, yAxisTitles, dateSelId) {{
  FIG_CONFIGS[divId] = {{zones: zonesList, dates: datesList, tracesPerCombo: tracesPerCombo, titlePrefix: titlePrefix, yAxisTitles: yAxisTitles || null, dateSelId: dateSelId || 'global-date'}};
}}

function applyFigSelection(divId) {{
  const cfg = FIG_CONFIGS[divId];
  const zoneSel = document.getElementById(divId + '-zone');
  const dateSel = document.getElementById(cfg.dateSelId);
  const zoneIdx = cfg.zones.indexOf(zoneSel.value);
  const dateIdx = cfg.dates.indexOf(dateSel.value);
  const total = cfg.zones.length * cfg.dates.length * cfg.tracesPerCombo;
  const visible = new Array(total).fill(false);
  const base = (zoneIdx * cfg.dates.length + dateIdx) * cfg.tracesPerCombo;
  for (let k = 0; k < cfg.tracesPerCombo; k++) visible[base + k] = true;
  Plotly.restyle(divId, {{visible: visible}});
  const relayout = {{title: cfg.titlePrefix + ' - ' + zoneSel.value + ' (' + dateSel.value + ')'}};
  if (cfg.yAxisTitles) relayout['yaxis.title'] = cfg.yAxisTitles[zoneIdx];
  Plotly.relayout(divId, relayout);
}}

function applyAllFigs() {{
  Object.keys(FIG_CONFIGS).forEach(applyFigSelection);
}}

function copyTableTSV(btn, dataObj, plotlyDivId) {{
  const gd = document.getElementById(plotlyDivId);
  const activeIdx = (gd && gd._fullLayout && gd._fullLayout.updatemenus && gd._fullLayout.updatemenus.length)
    ? gd._fullLayout.updatemenus[0].active : 0;
  const key = Object.keys(dataObj)[activeIdx];
  const d = dataObj[key];
  if (!d) return;
  const sep = '\\t';
  const header = 'Date' + sep + d.hours.map(h => 'H' + h).join(sep);
  const rows = d.dates.map((date, i) =>
    date + sep + d.values[i].map(v => v == null ? '' : v).join(sep)
  );
  navigator.clipboard.writeText([header, ...rows].join('\\n'))
    .then(() => {{ btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 1800); }})
    .catch(() => {{ btn.textContent = 'Failed';  setTimeout(() => btn.textContent = 'Copy', 1800); }});
}}
</script>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('dam', this)">Day-Ahead Market</button>
  <button class="tab-btn" onclick="showTab('rtm', this)">Real-Time Market</button>
  <button class="tab-btn" onclick="showTab('spread', this)">Spread</button>
  <button class="tab-btn" onclick="showTab('weather', this)">Weather</button>
  <button class="tab-btn" onclick="showTab('load', this)">Load Forecast</button>
  <button class="tab-btn" onclick="showTab('wind', this)">Wind Forecast</button>
  {dam_forecast_tab_button}
  {rtm_forecast_tab_button}
  {spread_forecast_tab_button}
</div>

<div class="global-day-bar">
  <label><strong>Day</strong> (applies to DAM, RTM, Spread, Load Forecast and Wind Forecast):</label>
  <select id="global-date" onchange="applyAllFigs()">
    {date_options_html(DAY_OPTION_STRS[default_date_idx])}
  </select>
</div>

<div id="tab-dam" class="tab-content active">
<div>
  <a class="refresh-btn" href="https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/dashboard.yml" target="_blank" rel="noopener">
    Refresh DAM (opens GitHub Actions)
  </a>
</div>
<h2>DAM - Hourly Profile</h2>
{zone_control('dam-hourly')}
{dam_hourly_fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='dam-hourly')}
{register_fig('dam-hourly', 3, 'DAM - Hourly Profile')}
<h2>DAM - Hourly Table (last {TABLE_DAYS} days) <button class="copy-btn" onclick="copyTableTSV(this, DAM_TABLE_DATA, 'dam-table')">Copy</button></h2>
{dam_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='dam-table')}
</div>

<div id="tab-rtm" class="tab-content">
<div>
  <a class="refresh-btn" href="https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/refresh_rtm.yml" target="_blank" rel="noopener">
    Refresh RTM (opens GitHub Actions)
  </a>
</div>
<h2>RTM - Hourly Profile</h2>
{zone_control('rtm-hourly')}
{rtm_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='rtm-hourly')}
{register_fig('rtm-hourly', 3, 'RTM - Hourly Profile')}
<h2>RTM - Hourly Table (last {TABLE_DAYS} days) <button class="copy-btn" onclick="copyTableTSV(this, RTM_TABLE_DATA, 'rtm-table')">Copy</button></h2>
{rtm_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='rtm-table')}
</div>

<div id="tab-spread" class="tab-content">
<h2>Spread (DAM - RTM) - Hourly Profile (Positive = green, Negative = red)</h2>
{zone_control('spread-hourly')}
{spread_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='spread-hourly')}
{register_fig('spread-hourly', 3, 'Spread (DAM - RTM) - Hourly Profile')}
<h2>Spread (DAM - RTM) - Hourly Table (last {TABLE_DAYS} days) <button class="copy-btn" onclick="copyTableTSV(this, SPREAD_TABLE_DATA, 'spread-table')">Copy</button></h2>
{spread_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='spread-table')}
</div>

<div id="tab-weather" class="tab-content">
<div>
  <a class="refresh-btn" href="https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/dashboard.yml" target="_blank" rel="noopener">
    Refresh Weather (opens GitHub Actions)
  </a>
</div>
<h2>Weather (OTTAWA) - Hourly Profile</h2>
{weather_control('weather-hourly')}
{weather_date_control('weather-hourly')}
{weather_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='weather-hourly')}
{register_fig('weather-hourly', 3, 'Weather - Hourly Profile', zones_json=WEATHER_LABELS_JSON, y_axis_titles=WEATHER_Y_AXIS_TITLES, date_sel_id='weather-hourly-date')}
<h2>Weather (OTTAWA) - Hourly Table (last {TABLE_DAYS} days) <button class="copy-btn" onclick="copyTableTSV(this, WEATHER_TABLE_DATA, 'weather-table')">Copy</button></h2>
{weather_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='weather-table')}
</div>

<div id="tab-load" class="tab-content">
<div>
  <a class="refresh-btn" href="https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/dashboard.yml" target="_blank" rel="noopener">
    Refresh Load Forecast (opens GitHub Actions)
  </a>
</div>
<h2>Load Forecast - Hourly Profile</h2>
{load_control('load-hourly')}
{load_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='load-hourly')}
{register_fig('load-hourly', 3, 'Load Forecast - Hourly Profile', zones_json=LOAD_LABELS_JSON, y_axis_titles=LOAD_Y_AXIS_TITLES)}
<h2>Load Forecast - Hourly Table (last {TABLE_DAYS} days) <button class="copy-btn" onclick="copyTableTSV(this, LOAD_TABLE_DATA, 'load-table')">Copy</button></h2>
{load_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='load-table')}
</div>

<div id="tab-wind" class="tab-content">
<div>
  <a class="refresh-btn" href="https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/dashboard.yml" target="_blank" rel="noopener">
    Refresh Wind Forecast (opens GitHub Actions)
  </a>
</div>
<h2>Wind Forecast - Hourly Profile</h2>
{wind_control('wind-hourly')}
{wind_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='wind-hourly')}
{register_fig('wind-hourly', 3, 'Wind Forecast - Hourly Profile', zones_json=WIND_ZONES_JSON)}
<h2>Wind Forecast - Hourly Table (last {TABLE_DAYS} days) <button class="copy-btn" onclick="copyTableTSV(this, WIND_TABLE_DATA, 'wind-table')">Copy</button></h2>
{wind_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='wind-table')}
</div>
{dam_forecast_tab_html}
{rtm_forecast_tab_html}
{spread_forecast_tab_html}
<script>
{dam_table_data_js}
{rtm_table_data_js}
{spread_table_data_js}
{weather_table_data_js}
{load_table_data_js}
{wind_table_data_js}
</script>

<footer>DAM data through: {latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; RTM data through: {rtm_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Weather data through: {weather_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Load Forecast through: {load_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Wind Forecast through: {wind_latest_ts.strftime('%Y-%m-%d %H:%M')}</footer>

</body>
</html>
"""

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Dashboard generated successfully at docs/index.html!")
