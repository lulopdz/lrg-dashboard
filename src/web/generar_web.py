"""Generate the main LRG dashboard page at docs/index.html."""

import json
import os

from dashboard_data import (
    COLORS,
    DAY_OPTION_STRS,
    DEFAULT_ZONE,
    FORECAST_ZONES,
    LOAD_VARS,
    TABLE_DAYS,
    default_date_idx,
    default_forecast_date_idx,
    default_load_idx,
    default_weather_idx,
    default_wind_idx,
    latest_ts,
    load_forecast,
    load_latest_ts,
    load_var_keys,
    rtm_latest_ts,
    weather_label,
    weather_latest_ts,
    weather_var_keys,
    wind_forecast,
    wind_latest_ts,
    wind_zones,
    zones,
)
from dashboard_figures import (
    build_hourly_fig,
    build_table_fig,
    build_wide_hourly_fig,
    build_wide_table_fig,
    compact_combos,
)
from gate import SITE_LOCK, gate_html
from refresh import DAILY_WORKFLOW, REFRESH_JS, RT_WORKFLOW, refresh_target
from tabs.common import (
    LOAD_LABELS_JSON,
    LOAD_Y_AXIS_TITLES,
    WIND_ZONES_JSON,
    date_input_html,
    register_fig,
    zone_options_html,
)
from tabs.market import build_market_tabs
from tabs.predict import build_predict_tabs
from tabs.supply import build_supply_tab
from tabs.weather import build_weather_tab
from theme import TIER_COLORS

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')


def _load_asset(filename, replacements):
    with open(os.path.join(ASSETS_DIR, filename), encoding='utf-8') as f:
        content = f.read().rstrip('\n')
    for token, value in replacements.items():
        content = content.replace(token, value)
    return content


def main():
    market_tabs_html, market_combos, COMPACT_ZONE_DATA_JSON = build_market_tabs()
    weather_tab_html, weather_combos, WEATHER_TILE_DATA_JSON = build_weather_tab()

    load_hourly_fig = build_wide_hourly_fig(
        load_forecast, 'interval_start_local', LOAD_VARS, default_load_idx, 'Load Forecast',
        default_day_idx=default_forecast_date_idx, polished=True,
    )
    load_table_fig = build_wide_table_fig(
        load_forecast, 'interval_start_local', LOAD_VARS, default_load_idx, 'Load Forecast',
        colorscale='Viridis', polished=True,
    )

    (
        supply_mix_tab_button,
        supply_mix_tab_html,
        supply_combos,
        SUPPLY_TILE_DATA_JSON,
        SUPPLY_TILE_LABELS_JSON,
        has_supply,
    ) = build_supply_tab()

    wind_hourly_fig = build_hourly_fig(
        wind_forecast, 'Wind Forecast', location_col='zone', value_col='generation_forecast',
        y_axis_title='Generation (MW)', zones_list=wind_zones, default_zone_idx=default_wind_idx,
        default_day_idx=default_forecast_date_idx, polished=True,
    )
    wind_table_fig = build_table_fig(
        wind_forecast, 'Wind Forecast', palette='Greens', location_col='zone',
        value_col='generation_forecast', zones_list=wind_zones, default_zone_idx=default_wind_idx,
        colorbar_title='MW', hover_label='Generation', hover_prefix='', hover_suffix=' MW',
        polished=True,
    )

    # Every chart that follows the Day picker is built as one hidden trace group per (zone, day);
    # keep only the visible group and ship the rest as data (see compact_combos / showCombo).
    # Keys are the charts' div ids. The DAM/RTM/Spread charts also carry 3 trailing 'dynamic'
    # traces for non-default zones (compact_zones=True), which stay as they are.
    COMBO_DATA = {
        **market_combos,
        'load-hourly': compact_combos(load_hourly_fig, 3),
        'wind-hourly': compact_combos(wind_hourly_fig, 3),
        **weather_combos,
        **supply_combos,
    }
    COMBO_DATA_JSON = json.dumps(COMBO_DATA, separators=(',', ':'))

    predict = build_predict_tabs()
    dam_forecast_tab_button = predict['dam_forecast_tab_button']
    dam_forecast_tab_html = predict['dam_forecast_tab_html']
    rtm_forecast_tab_button = predict['rtm_forecast_tab_button']
    rtm_forecast_tab_html = predict['rtm_forecast_tab_html']
    spread_forecast_tab_button = predict['spread_forecast_tab_button']
    spread_forecast_tab_html = predict['spread_forecast_tab_html']

    # One refresh target per tab, swapped into the day-bar's single button by tab name (see
    # showTab in the page JS). Two modes only: RT and Spread refresh on demand (refresh_rtm.yml),
    # everything else comes from the 9:00 daily run (daily.yml).
    TAB_REFRESH = {
        'dam': (DAILY_WORKFLOW, 'Run daily update'),
        'rtm': (RT_WORKFLOW, 'Refresh Real-Time'),
        'spread': (RT_WORKFLOW, 'Refresh Real-Time'),
        'weather': (DAILY_WORKFLOW, 'Run daily update'),
        'load': (DAILY_WORKFLOW, 'Run daily update'),
        'wind': (DAILY_WORKFLOW, 'Run daily update'),
        'supply': (DAILY_WORKFLOW, 'Run daily update'),
    }
    for _tab, _btn in [('forecast', dam_forecast_tab_button), ('rtm-forecast', rtm_forecast_tab_button),
                       ('spread-signal', spread_forecast_tab_button)]:
        if _btn:
            TAB_REFRESH[_tab] = (DAILY_WORKFLOW, 'Run daily update')

    TAB_REFRESH_JSON = json.dumps({tab: refresh_target(workflow, label) for tab, (workflow, label) in TAB_REFRESH.items()})

    # Which tabs have a single dropdown filter unified into the shared day-bar selector (drives
    # the hourly chart and/or the table's trace visibility -- see applyZoneChange in the page
    # JS). Despite the name, it's generic: 'label'/'options' just relabel and repopulate the same
    # select, so it doubles as the Weather Forecast tab's Variable picker. Weather has no
    # 'hourlyDiv' -- its small-multiples grid shows every variable at once (see
    # build_weather_grid_figs), so only the table still needs a variable filter.
    TAB_ZONES = {
        'dam': {'label': 'Zone', 'options': zones, 'default': DEFAULT_ZONE, 'hourlyDiv': 'dam-hourly', 'tableDiv': 'dam-table'},
        'rtm': {'label': 'Zone', 'options': zones, 'default': DEFAULT_ZONE, 'hourlyDiv': 'rtm-hourly', 'tableDiv': 'rtm-table'},
        'spread': {'label': 'Zone', 'options': zones, 'default': DEFAULT_ZONE, 'hourlyDiv': 'spread-hourly', 'tableDiv': 'spread-table'},
        'weather': {
            'label': 'Variable',
            'options': [weather_label(v) for v in weather_var_keys],
            'default': weather_label(weather_var_keys[default_weather_idx]),
            'tableDiv': 'weather-table',
        },
        'load': {
            'label': 'Zone',
            'options': [LOAD_VARS[v][0] for v in load_var_keys],
            'default': LOAD_VARS[load_var_keys[default_load_idx]][0],
            'hourlyDiv': 'load-hourly', 'tableDiv': 'load-table',
        },
        'wind': {
            'label': 'Zone',
            'options': wind_zones, 'default': wind_zones[default_wind_idx],
            'hourlyDiv': 'wind-hourly', 'tableDiv': 'wind-table',
        },
    }

    # Each tab group opens on its own default day: the market tabs (DAM/RTM/Spread) on today,
    # the forecast tabs (Weather/Load/Wind, which all carry a look-ahead forecast) on tomorrow.
    # The figures themselves are already built with the matching trace pre-visible (see
    # default_day_idx above); this just keeps the shared day-bar's displayed value from
    # contradicting whatever the chart is actually showing when a tab is opened.
    TAB_DATES = {
        'dam': DAY_OPTION_STRS[default_date_idx],
        'rtm': DAY_OPTION_STRS[default_date_idx],
        'spread': DAY_OPTION_STRS[default_date_idx],
        'weather': DAY_OPTION_STRS[default_forecast_date_idx],
        'load': DAY_OPTION_STRS[default_forecast_date_idx],
        'wind': DAY_OPTION_STRS[default_forecast_date_idx],
    }
    if has_supply:
        TAB_DATES['supply'] = DAY_OPTION_STRS[default_forecast_date_idx]
    TAB_DATES.update(predict['tab_dates'])

    # The predict tabs get the day-bar's Zone select once there's more than one zone to pick;
    # 'panels' tells applyZoneChange to swap the tab's .zone-panel instead of restyling a chart.
    if len(FORECAST_ZONES) > 1:
        for _tab in ('forecast', 'rtm-forecast', 'spread-signal'):
            TAB_ZONES[_tab] = {'label': 'Zone', 'options': FORECAST_ZONES, 'default': FORECAST_ZONES[0], 'panels': True}
    TAB_ZONES_JSON = json.dumps(TAB_ZONES)
    TAB_DATES_JSON = json.dumps(TAB_DATES)

    dashboard_css = _load_asset('dashboard.css', {
        '/*__COLOR_RING__*/': COLORS['ring'],
        '/*__COLOR_MUTED__*/': COLORS['muted'],
        '/*__COLOR_DAM__*/': COLORS['dam'],
        '/*__COLOR_FORECAST__*/': COLORS['forecast'],
        '/*__COLOR_PREDICTED__*/': COLORS['predicted'],
        '/*__COLOR_POSITIVE__*/': COLORS['positive'],
        '/*__COLOR_NEGATIVE__*/': COLORS['negative'],
        '/*__TIER_POS_HIGH__*/': TIER_COLORS[('DART > 0', 'high')],
        '/*__TIER_NEG_HIGH__*/': TIER_COLORS[('DART < 0', 'high')],
    })
    dashboard_js = _load_asset('dashboard.js', {
        '/*__COLOR_MUTED__*/': COLORS['muted'],
        '/*__COLOR_POSITIVE__*/': COLORS['positive'],
        '/*__COLOR_NEGATIVE__*/': COLORS['negative'],
        '/*__TIER_COLORS_JSON__*/': predict['TIER_COLORS_JSON'],
    })

    # 4. Assemble the HTML page with tabs (DAM / RTM / Spread / Weather / Load Forecast / Wind Forecast)
    # and shared zone/day controls
    os.makedirs('docs', exist_ok=True)

    html = f"""<html>
<head>
<meta charset="utf-8">
<title>DAM Dashboard</title>
<style>
{dashboard_css}
</style>
</head>
<body>

{gate_html([SITE_LOCK])}

<script>
const TAB_REFRESH = {TAB_REFRESH_JSON};
const TAB_ZONES = {TAB_ZONES_JSON};
const TAB_DATES = {TAB_DATES_JSON};
// {{dam: {{zone: {{date: [24 hourly values]}}}}, rtm: {{...}}}} -- every zone except the default
// (OTTAWA), which build_hourly_fig(compact_zones=True) already pre-bakes as real Plotly
// traces. Read by showCompactZone() to recompute DAM/RTM on the fly when a non-default zone
// is picked, instead of shipping every zone x day combination pre-baked.
const COMPACT_ZONE_DATA = {COMPACT_ZONE_DATA_JSON};
const COMBO_DATA = {COMBO_DATA_JSON};
// Per-day Supply Mix tile values, so the tiles follow the Day picker like the charts do.
const SUPPLY_TILE_DATA = {SUPPLY_TILE_DATA_JSON};
const SUPPLY_TILE_LABELS = {SUPPLY_TILE_LABELS_JSON};
const WEATHER_TILE_DATA = {WEATHER_TILE_DATA_JSON};
// Archived forecasts per series and target day, with what actually cleared where known.
const FORECAST_HISTORY = {predict['FORECAST_HISTORY_JSON']};
const FORECAST_TABS = {predict['FORECAST_TABS_JSON']};
const FORECAST_LATEST = {predict['FORECAST_LATEST_JSON']};
const SIGNAL_HISTORY = {predict['SIGNAL_HISTORY_JSON']};
const SIGNAL_BASE = {predict['SIGNAL_BASE_JSON']};
// The zone the predict tabs show (DAM/RTM Forecast, Spread Signal): one for the three, so picking
// TORONTO on one keeps it on the others. Each zone's elements carry the id suffix '--' + zone.
let forecastZone = {predict['FORECAST_ZONE_JSON']};
const fz = base => base + '--' + forecastZone;
const DAY_RANGE = ['{DAY_OPTION_STRS[0]}', '{DAY_OPTION_STRS[-1]}'];
const forecastMarks = {{}};  // each forecast chart's original vline/annotation, restored on the live day
let currentTab = 'dam';
{REFRESH_JS}

{dashboard_js}
</script>

<div class="tabs-row">
  <div class="tabs">
    <button class="tab-btn group-market active" onclick="showTab('dam', this)">DAM</button>
    <button class="tab-btn group-market" onclick="showTab('rtm', this)">RTM</button>
    <button class="tab-btn group-market" onclick="showTab('spread', this)">Spread</button>
    <button class="tab-btn group-forecast tab-group-start" onclick="showTab('weather', this)">Weather Forecast</button>
    <button class="tab-btn group-forecast" onclick="showTab('load', this)">Load Forecast</button>
    <button class="tab-btn group-forecast" onclick="showTab('wind', this)">Wind Forecast</button>
    {supply_mix_tab_button}
    {dam_forecast_tab_button}
    {rtm_forecast_tab_button}
    {spread_forecast_tab_button}
  </div>
  <a href="portfolio.html" class="sim-link" title="View trading history and PnL" style="margin-right: 12px;">Portfolio</a>
  <a href="simulator.html" class="sim-link" title="Practice with historical backtests, no spoilers">Simulator</a>
</div>

<div class="global-day-bar">
  <div class="day-bar-left">
    <label><strong>Day</strong> (applies to DAM, RTM, Spread, the Weather/Load/Wind Forecasts, the DAM/RTM Forecast tabs and the Spread Signal):</label>
    {date_input_html('global-date', 'applyAllFigs()', DAY_OPTION_STRS[default_date_idx])}
    <div id="tab-zone-group" class="controls">
      <label id="tab-zone-label">Zone:</label>
      <select id="tab-zone-select" onchange="applyZoneChange()">
        {zone_options_html(zones, DEFAULT_ZONE)}
      </select>
    </div>
  </div>
  <a id="tab-refresh-btn" class="refresh-btn" href="{refresh_target(*TAB_REFRESH['dam'])['href']}"
     target="_blank" rel="noopener" title="Opens GitHub Actions">{TAB_REFRESH['dam'][1]}</a>
</div>

{market_tabs_html}

{weather_tab_html}

<div id="tab-load" class="tab-content">
<div class="card">
  <div class="section-header">
    <h2>Hourly Profile</h2>
  </div>
  {load_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='load-hourly')}
  {register_fig('load-hourly', 3, 'Load Forecast - Hourly Profile', zones_json=LOAD_LABELS_JSON, y_axis_titles=LOAD_Y_AXIS_TITLES, show_title=False, zone_sel_id='tab-zone-select')}
</div>
<div class="card">
  <div class="section-header">
    <h2>Hourly Table (last {TABLE_DAYS} days)</h2>
    <button class="copy-btn" onclick="copyTableTSV(this, 'load-table')">Copy</button>
  </div>
  {load_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='load-table')}
  <script>registerTable('load-table', {LOAD_LABELS_JSON});</script>
</div>
</div>

<div id="tab-wind" class="tab-content">
<div class="card">
  <div class="section-header">
    <h2>Hourly Profile</h2>
  </div>
  {wind_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='wind-hourly')}
  {register_fig('wind-hourly', 3, 'Wind Forecast - Hourly Profile', zones_json=WIND_ZONES_JSON, show_title=False, zone_sel_id='tab-zone-select')}
</div>
<div class="card">
  <div class="section-header">
    <h2>Hourly Table (last {TABLE_DAYS} days)</h2>
    <button class="copy-btn" onclick="copyTableTSV(this, 'wind-table')">Copy</button>
  </div>
  {wind_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='wind-table')}
  <script>registerTable('wind-table', {WIND_ZONES_JSON});</script>
</div>
</div>

{supply_mix_tab_html}
{dam_forecast_tab_html}
{rtm_forecast_tab_html}
{spread_forecast_tab_html}

<footer>DAM data through: {latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; RTM data through: {rtm_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Weather Forecast data through: {weather_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Load Forecast through: {load_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Wind Forecast through: {wind_latest_ts.strftime('%Y-%m-%d %H:%M')}</footer>

</body>
</html>
"""

    # Defer drawing to the first time each chart's tab is shown (see lazyPlot in the page script).
    # Only Plotly's own figure scripts call newPlot; the page's code above only restyles.
    html = html.replace('Plotly.newPlot(', 'lazyPlot(')

    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("Dashboard generated successfully at docs/index.html!")


if __name__ == '__main__':
    main()
