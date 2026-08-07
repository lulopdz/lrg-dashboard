"""Builds the Trading Simulator: pick a past date D, and the page shows a dashboard-like
view restricted to what would actually have been known at that moment (not a live replica of
the main dashboard's data -- a separate, self-contained restriction of it):

  - DAM: the previous day (D-1), fully settled.
  - RTM: the chosen day (D), but only through HE08 (when trades normally get decided).
  - Spread (DAM - RTM): the previous day (D-1), since it needs both sides settled.
  - Weather / Load / Wind Forecast: the outlook for the next day (D+1).

After reviewing that, a "Ready? Go to simulator" button reveals the same Long/Flat/Short
per-hour evaluation game as before: place a call on the DAM-RTM spread for each hour of D,
then reveal the actual outcome and score against the best-possible P&L that day.

Fully client-side: no backend, no server-side scoring, no data-vintage enforcement (see the
known limitation below). The history of past attempts lives in the browser's localStorage.
The restriction above is a *view* restriction, not a hard data cut: the full history is
embedded on the page (like the rest of this site -- fetch() against a local file fails under
file://, which is how this site gets tested before every push), so it depends on the user's
own discipline not to peek ahead of what a tab is meant to show, the same tradeoff already
accepted for the rest of the simulator's history.

Known limitation: the weather/load/wind forecast CSVs get overwritten on every daily refresh
and don't preserve the forecast as it looked at the time, so the D+1 "forecast" shown here is
actually the realized historical data for that date, not necessarily what was forecast back
then. Accepted for now; fixing it needs a separate daily-vintage archive (see conversation).

A separate page from the main dashboard (see generar_web.py), which stays untouched.
Outputs docs/simulator.html (data embedded inline, see SIM_DATA_JSON below).
"""
import json
import os

import pandas as pd

DEFAULT_ZONE = 'OTTAWA'
WIND_ZONE = 'Ontario Total'
CONTEXT_DAYS = 14  # minimum trailing history required before a date becomes playable
TRADING_CUTOFF_HOUR = 8  # HE08 -- RTM context for the chosen day stops here

WEATHER_VARS = {
    "temperature_2m": ("Temperature", "°C"),
    "wind_speed_10m": ("Wind Speed", "m/s"),
    "precipitation": ("Precipitation", "mm"),
    "snowfall": ("Snowfall", "cm"),
    "shortwave_radiation": ("Solar Radiation", "W/m²"),
    "relative_humidity_2m": ("Humidity", "%"),
}
LOAD_VARS = {
    "ontario": ("Ontario (Total)", "MW"),
    "ontario_northeast": ("Northeast", "MW"),
    "ontario_northwest": ("Northwest", "MW"),
    "ontario_southwest": ("Southwest", "MW"),
    "ontario_southeast": ("Southeast", "MW"),
}

os.makedirs('docs', exist_ok=True)


def hourly_map(df, date_col, value_col):
    """date -> [24 hourly values, None where missing]."""
    df = df.copy()
    df['hour'] = df[date_col].dt.hour + 1
    df['date'] = df[date_col].dt.date.astype(str)
    return {
        date: [None if pd.isna(v) else round(float(v), 2)
               for v in g.set_index('hour')[value_col].reindex(range(1, 25))]
        for date, g in df.groupby('date')
    }


dam_csv = pd.read_csv('data/ieso_dam_prices.csv', parse_dates=['interval_start_local'])
dam_by_date = hourly_map(dam_csv[dam_csv['location'] == DEFAULT_ZONE], 'interval_start_local', 'lmp')

rtm_csv = pd.read_csv('data/ieso_rtm_prices.csv', parse_dates=['interval_start_local'])
rtm_by_date = hourly_map(rtm_csv[rtm_csv['location'] == DEFAULT_ZONE], 'interval_start_local', 'lmp')

wind_csv = pd.read_csv('data/ieso_wind_forecast.csv', parse_dates=['interval_start_local'])
wind_by_date = hourly_map(wind_csv[wind_csv['zone'] == WIND_ZONE], 'interval_start_local', 'generation_forecast')

weather_csv = pd.read_csv('data/OTTAWA_weather.csv', parse_dates=['timestamp'])
weather_by_var = {var: hourly_map(weather_csv, 'timestamp', var) for var in WEATHER_VARS}

load_csv = pd.read_csv('data/ieso_load_forecast.csv', parse_dates=['interval_start_local'])
load_by_var = {var: hourly_map(load_csv, 'interval_start_local', var) for var in LOAD_VARS}

all_dates = sorted(set(dam_by_date) | set(rtm_by_date))

# Today (and, for DAM, tomorrow) aren't fully realized yet -- the latest playable date is
# yesterday, in the same market-time convention dashboard_data.py uses for today_date.
today_date = pd.Timestamp.now(tz='UTC').tz_convert('-05:00').date()
max_playable = str(today_date - pd.Timedelta(days=1))
playable_dates = [d for d in all_dates if d <= max_playable]
min_playable = playable_dates[CONTEXT_DAYS] if len(playable_dates) > CONTEXT_DAYS else playable_dates[0]
max_playable = playable_dates[-1] if playable_dates else max_playable

data = {
    'zone': DEFAULT_ZONE,
    'windZone': WIND_ZONE,
    'dates': all_dates,
    'dam': dam_by_date,
    'rtm': rtm_by_date,
    'wind': wind_by_date,
    'weather': weather_by_var,
    'load': load_by_var,
    'weatherLabels': {var: {'label': label, 'unit': unit} for var, (label, unit) in WEATHER_VARS.items()},
    'loadLabels': {var: {'label': label, 'unit': unit} for var, (label, unit) in LOAD_VARS.items()},
}
# Embedded inline in the page (like every other figure/table on the main dashboard) instead
# of a separate fetch()'d JSON file: fetch() against a local file fails under file:// (CORS),
# which is how this site gets tested before every push -- no fetch calls anywhere else in
# this codebase for the same reason.
SIM_DATA_JSON = json.dumps(data)


html = f"""<html>
<head>
<meta charset="utf-8">
<title>Trading Simulator</title>
<style>
  body {{ background:#111; color:#eee; margin:0; padding:24px;
    font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  a {{ color:#3498db; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:15px; color:#ddd; margin:0 0 10px; }}
  h3 {{ margin:0 0 4px; font-size:13px; color:#ccc; font-weight:600; }}
  .subtitle {{ color:#888; font-size:13px; margin:0 0 20px; max-width:760px; }}
  .back-link {{ display:inline-block; margin-bottom:16px; color:#aaa; text-decoration:none; font-size:13px; }}
  .back-link:hover {{ color:#fff; }}
  .card {{ background:#161616; border:1px solid #2a2a2a; border-radius:8px; padding:16px 20px; margin:16px 0; }}
  .controls-row {{ display:flex; align-items:center; gap:16px; flex-wrap:wrap; }}
  select, input[type="date"] {{ background:#1e1e1e; color:#eee; border:1px solid #444; border-radius:4px; padding:5px 9px; font-family:inherit; }}
  input[type="date"]::-webkit-calendar-picker-indicator {{ filter:invert(0.8); }}
  button.primary {{ background:#9b59b6; border:none; border-radius:4px; color:#fff; cursor:pointer; padding:8px 18px; font-size:14px; }}
  button.primary:hover {{ background:#8e44ad; }}
  button.secondary {{ background:#2c2c2c; border:1px solid #444; border-radius:4px; color:#aaa; cursor:pointer; padding:6px 14px; font-size:12px; }}
  button.secondary:hover {{ color:#fff; }}
  .stat-row {{ display:flex; flex-wrap:wrap; gap:16px; margin:16px 0; }}
  .stat-tile {{ background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:12px 20px; }}
  .stat-label {{ color:#888; font-size:12px; }}
  .stat-value {{ color:#eee; font-size:22px; font-weight:600; margin-top:4px; }}
  .hour-strip {{ display:flex; gap:6px; overflow-x:auto; padding-bottom:8px; }}
  .hour-cell {{ flex:0 0 auto; width:44px; text-align:center; }}
  .hour-cell .hour-label {{ color:#888; font-size:11px; margin-bottom:4px; }}
  .pos-btn {{ display:block; width:100%; border:1px solid #333; background:#1e1e1e; color:#777;
    font-size:12px; padding:4px 0; cursor:pointer; margin-bottom:2px; }}
  .pos-btn:first-child {{ border-radius:4px 4px 0 0; }}
  .pos-btn:last-child {{ border-radius:0 0 4px 4px; margin-bottom:0; }}
  .pos-btn.active.long {{ background:#2ecc71; color:#111; border-color:#2ecc71; }}
  .pos-btn.active.flat {{ background:#666; color:#fff; border-color:#666; }}
  .pos-btn.active.short {{ background:#e74c3c; color:#111; border-color:#e74c3c; }}
  table.history-table {{ border-collapse:collapse; width:100%; }}
  table.history-table th, table.history-table td {{ padding:6px 12px; text-align:right; border-bottom:1px solid #333; font-size:13px; }}
  table.history-table th:first-child, table.history-table td:first-child {{ text-align:left; }}
  table.history-table th {{ color:#888; font-weight:500; }}
  table.history-table tfoot td {{ color:#ccc; font-weight:600; border-bottom:none; padding-top:8px; }}
  .hidden {{ display:none; }}
  .disclaimer {{ color:#666; font-size:11px; margin-top:12px; }}
  .tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin:0 0 -1px; }}
  .tab-btn {{
    background:#1e1e1e; color:#ccc; border:1px solid #333; border-top:3px solid transparent;
    border-radius:6px 6px 0 0; padding:8px 16px; cursor:pointer; font-size:13px;
  }}
  .tab-btn.active {{ background:#161616; color:#fff; border-bottom:2px solid #666; }}
  .tab-btn.group-market {{ border-top-color:#3498db; }}
  .tab-btn.group-market.active {{ border-bottom:2px solid #3498db; }}
  .tab-btn.group-forecast {{ border-top-color:#14b8a6; }}
  .tab-btn.group-forecast.active {{ border-bottom:2px solid #14b8a6; }}
  .rtab-content {{ max-height:0; overflow:hidden; }}
  .rtab-content.active {{ max-height:none; }}
  .day-label {{ color:#888; font-size:12px; margin-bottom:8px; }}
  .mini-grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; }}
  @media (max-width: 900px) {{ .mini-grid {{ grid-template-columns:repeat(2, 1fr); }} }}
  @media (max-width: 600px) {{ .mini-grid {{ grid-template-columns:1fr; }} }}
  .mini-tile {{ background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px; padding:12px 14px; }}
</style>
</head>
<body>

<a class="back-link" href="index.html">&larr; Back to dashboard</a>
<h1>Trading Simulator</h1>
<p class="subtitle">Pick a past date and this page shows only what would have been known at
that moment: DAM shows the previous day (D&minus;1, fully settled), RTM shows the chosen day
(D) through HE{TRADING_CUTOFF_HOUR:02d} (normally when trades get decided), and Weather / Load
/ Wind Forecast show the outlook for the next day (D+1). Review it, then place your Long / Flat
/ Short call per hour and reveal how you did against the actual DAM&minus;RTM spread. Zone:
{DEFAULT_ZONE} (Wind uses {WIND_ZONE}). Known limitation: weather/load/wind forecasts get
overwritten every day and don't preserve what was actually known at the time, so D+1 here is
the realized historical data for that date, not necessarily the exact forecast as it looked
back then.</p>

<div class="card">
  <div class="controls-row">
    <label><strong>Date</strong> to simulate:</label>
    <input type="date" id="sim-date" min="{min_playable}" max="{max_playable}" value="{max_playable}">
    <button class="primary" onclick="loadDay()">Load day</button>
    <span id="load-status" style="color:#888; font-size:13px;"></span>
  </div>
</div>

<div id="dashboard-section" class="hidden">
  <div class="tabs">
    <button class="tab-btn group-market active" onclick="showRTab('dam', this)">DAM</button>
    <button class="tab-btn group-market" onclick="showRTab('rtm', this)">RTM</button>
    <button class="tab-btn group-market" onclick="showRTab('spread', this)">Spread</button>
    <button class="tab-btn group-forecast" onclick="showRTab('weather', this)">Weather Forecast</button>
    <button class="tab-btn group-forecast" onclick="showRTab('load', this)">Load Forecast</button>
    <button class="tab-btn group-forecast" onclick="showRTab('wind', this)">Wind Forecast</button>
  </div>

  <div id="rtab-dam" class="rtab-content card active">
    <div class="day-label" id="dam-day-label"></div>
    <div id="dam-chart" style="height:300px;"></div>
  </div>
  <div id="rtab-rtm" class="rtab-content card">
    <div class="day-label" id="rtm-day-label"></div>
    <div id="rtm-chart" style="height:300px;"></div>
  </div>
  <div id="rtab-spread" class="rtab-content card">
    <div class="day-label" id="spread-day-label"></div>
    <div id="spread-chart" style="height:300px;"></div>
  </div>
  <div id="rtab-weather" class="rtab-content card">
    <div class="day-label" id="weather-day-label"></div>
    <div class="mini-grid" id="weather-grid"></div>
  </div>
  <div id="rtab-load" class="rtab-content card">
    <div class="day-label" id="load-day-label"></div>
    <div class="mini-grid" id="load-grid"></div>
  </div>
  <div id="rtab-wind" class="rtab-content card">
    <div class="day-label" id="wind-day-label"></div>
    <div id="wind-chart" style="height:300px;"></div>
  </div>

  <button class="primary" onclick="goToEvaluation()" style="margin:8px 0 20px;">Ready? Go to simulator</button>
</div>

<div id="entry-section" class="card hidden">
  <h2>Your position per hour (Long / Flat / Short DAM&minus;RTM)</h2>
  <div class="hour-strip" id="hour-strip"></div>
  <button class="primary" onclick="reveal()" style="margin-top:12px;">Reveal result</button>
</div>

<div id="result-section" class="card hidden">
  <h2>Result</h2>
  <div class="stat-row" id="result-stats"></div>
  <div id="result-chart" style="height:320px;"></div>
</div>

<div class="card">
  <div class="controls-row" style="justify-content:space-between;">
    <h2 style="margin:0;">Your history</h2>
    <button class="secondary" onclick="clearHistory()">Clear history</button>
  </div>
  <table class="history-table" id="history-table"></table>
  <p class="disclaimer">The real data for every date is already loaded in your browser
  (this is a static site, with no server), so what each tab shows you is a matter of your own
  discipline not to look further than intended, not a technical restriction. Your history is
  saved only in this browser (localStorage): it does not sync across devices or upload anywhere.</p>
</div>

<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<script>
const SIM_ZONE = {json.dumps(DEFAULT_ZONE)};
const SIM_WIND_ZONE = {json.dumps(WIND_ZONE)};
const TRADING_CUTOFF_HOUR = {TRADING_CUTOFF_HOUR};
const SIM_DATA = {SIM_DATA_JSON};
let currentDate = null;
let positions = {{}};

function addDays(dateStr, n) {{
  const d = new Date(dateStr + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}}

function sevenDayAvg(seriesMap, anchorDate) {{
  const sum = new Array(24).fill(0), count = new Array(24).fill(0);
  for (let i = 1; i <= 7; i++) {{
    const row = seriesMap[addDays(anchorDate, -i)];
    if (!row) continue;
    for (let h = 0; h < 24; h++) {{ if (row[h] != null) {{ sum[h] += row[h]; count[h]++; }} }}
  }}
  return sum.map((s, h) => count[h] ? s / count[h] : null);
}}

function renderProfileChart(divId, traces, yTitle, compact) {{
  const hours = Array.from({{length: 24}}, (_, i) => i + 1);
  const plotData = traces.map(t => ({{
    x: hours, y: t.y, name: t.name, mode: t.marker === false ? 'lines' : 'lines+markers',
    line: {{color: t.color, dash: t.dash || 'solid', width: t.width || 2.5}},
    marker: t.marker === false ? undefined : {{size: compact ? 5 : 7, line: {{width: 1, color: '#111'}}}}
  }}));
  Plotly.newPlot(divId, plotData, {{
    template: 'plotly_dark', paper_bgcolor: compact ? '#1a1a1a' : '#161616', plot_bgcolor: compact ? '#1a1a1a' : '#161616',
    margin: compact ? {{t: 6, b: 26, l: 40, r: 8}} : {{t: 10, b: 40, l: 55, r: 20}},
    xaxis: {{title: compact ? '' : 'Hour', dtick: compact ? 4 : 1, range: [0.5, 24.5], gridcolor: '#242424'}},
    yaxis: {{title: compact ? '' : yTitle, gridcolor: '#242424', hoverformat: '.1f'}},
    hovermode: 'x unified',
    showlegend: !compact,
    legend: compact ? undefined : {{orientation: 'h', y: 1.15}}
  }}, {{displayModeBar: false, responsive: true}});
}}

function showRTab(name, btn) {{
  document.querySelectorAll('.rtab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tabs .tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('rtab-' + name).classList.add('active');
  btn.classList.add('active');
}}

function loadDay() {{
  const date = document.getElementById('sim-date').value;
  if (!date) return;
  if (!SIM_DATA.dates.includes(date)) {{
    document.getElementById('load-status').textContent = 'No data for that date.';
    return;
  }}
  document.getElementById('load-status').textContent = '';
  currentDate = date;
  positions = {{}};
  for (let h = 1; h <= 24; h++) positions[h] = 'flat';

  document.getElementById('result-section').classList.add('hidden');
  document.getElementById('entry-section').classList.add('hidden');
  document.getElementById('dashboard-section').classList.remove('hidden');

  const prevDate = addDays(date, -1);
  const nextDate = addDays(date, 1);

  renderDamTab(prevDate);
  renderRtmTab(date);
  renderSpreadTab(prevDate);
  renderWeatherTab(nextDate);
  renderLoadTab(nextDate);
  renderWindTab(nextDate);
  renderHourStrip();
}}

function renderDamTab(damDate) {{
  document.getElementById('dam-day-label').textContent = 'Showing: ' + damDate + ' (previous day, D-1)';
  const avg = sevenDayAvg(SIM_DATA.dam, damDate);
  const main = SIM_DATA.dam[damDate] || new Array(24).fill(null);
  renderProfileChart('dam-chart', [
    {{y: avg, name: '7d Average', color: '#6b7280', dash: 'dot', width: 1.5, marker: false}},
    {{y: main, name: damDate, color: '#3498db', width: 3}}
  ], 'DAM Price ($/MWh)', false);
}}

function renderRtmTab(rtmDate) {{
  const cutoffLabel = 'HE' + String(TRADING_CUTOFF_HOUR).padStart(2, '0');
  document.getElementById('rtm-day-label').textContent = 'Showing: ' + rtmDate + ' (chosen day D, through ' + cutoffLabel + ')';
  const avg = sevenDayAvg(SIM_DATA.rtm, rtmDate);
  const rowFull = SIM_DATA.rtm[rtmDate] || new Array(24).fill(null);
  const masked = rowFull.map((v, i) => i < TRADING_CUTOFF_HOUR ? v : null);
  renderProfileChart('rtm-chart', [
    {{y: avg, name: '7d Average', color: '#6b7280', dash: 'dot', width: 1.5, marker: false}},
    {{y: masked, name: rtmDate, color: '#e67e22', width: 3}}
  ], 'RTM Price ($/MWh)', false);
}}

function renderSpreadTab(spreadDate) {{
  document.getElementById('spread-day-label').textContent = 'Showing: ' + spreadDate + ' (previous day, D-1)';
  const hours = Array.from({{length: 24}}, (_, i) => i + 1);
  const dam = SIM_DATA.dam[spreadDate] || new Array(24).fill(null);
  const rtm = SIM_DATA.rtm[spreadDate] || new Array(24).fill(null);
  const spread = dam.map((d, i) => (d != null && rtm[i] != null) ? d - rtm[i] : null);
  const colors = spread.map(v => v == null ? '#444' : (v >= 0 ? '#2ecc71' : '#e74c3c'));
  Plotly.newPlot('spread-chart', [{{
    x: hours, y: spread, type: 'bar', marker: {{color: colors}},
    hovertemplate: 'Hour %{{x}}<br>Spread: $%{{y:.1f}}<extra></extra>'
  }}], {{
    template: 'plotly_dark', paper_bgcolor: '#161616', plot_bgcolor: '#161616',
    margin: {{t: 10, b: 40, l: 50, r: 20}},
    xaxis: {{title: 'Hour', dtick: 1, gridcolor: '#242424'}},
    yaxis: {{title: 'Spread (DAM - RTM, $/MWh)', gridcolor: '#242424', hoverformat: '.1f', zerolinecolor: '#666'}}
  }}, {{displayModeBar: false, responsive: true}});
}}

function renderMiniGrid(gridId, varMap, seriesLookup, anchorDate) {{
  const grid = document.getElementById(gridId);
  grid.innerHTML = '';
  Object.keys(varMap).forEach((varKey, i) => {{
    const meta = varMap[varKey];
    const divId = gridId + '-tile-' + i;
    const tile = document.createElement('div');
    tile.className = 'mini-tile';
    const h3 = document.createElement('h3');
    h3.textContent = meta.unit ? (meta.label + ' (' + meta.unit + ')') : meta.label;
    const chartDiv = document.createElement('div');
    chartDiv.id = divId;
    chartDiv.style.height = '220px';
    tile.appendChild(h3);
    tile.appendChild(chartDiv);
    grid.appendChild(tile);

    const seriesMap = seriesLookup[varKey];
    const avg = sevenDayAvg(seriesMap, anchorDate);
    const main = seriesMap[anchorDate] || new Array(24).fill(null);
    renderProfileChart(divId, [
      {{y: avg, name: '7d Average', color: '#6b7280', dash: 'dot', width: 1.5, marker: false}},
      {{y: main, name: anchorDate, color: '#14b8a6', width: 2.5}}
    ], null, true);
  }});
}}

function renderWeatherTab(weatherDate) {{
  document.getElementById('weather-day-label').textContent = 'Showing forecast for: ' + weatherDate + ' (next day, D+1)';
  renderMiniGrid('weather-grid', SIM_DATA.weatherLabels, SIM_DATA.weather, weatherDate);
}}

function renderLoadTab(loadDate) {{
  document.getElementById('load-day-label').textContent = 'Showing forecast for: ' + loadDate + ' (next day, D+1)';
  renderMiniGrid('load-grid', SIM_DATA.loadLabels, SIM_DATA.load, loadDate);
}}

function renderWindTab(windDate) {{
  document.getElementById('wind-day-label').textContent = 'Showing forecast for: ' + windDate + ' (next day, D+1), zone: ' + SIM_WIND_ZONE;
  const avg = sevenDayAvg(SIM_DATA.wind, windDate);
  const main = SIM_DATA.wind[windDate] || new Array(24).fill(null);
  renderProfileChart('wind-chart', [
    {{y: avg, name: '7d Average', color: '#6b7280', dash: 'dot', width: 1.5, marker: false}},
    {{y: main, name: windDate, color: '#14b8a6', width: 3}}
  ], 'Generation (MW)', false);
}}

function goToEvaluation() {{
  const section = document.getElementById('entry-section');
  section.classList.remove('hidden');
  section.scrollIntoView({{behavior: 'smooth', block: 'start'}});
}}

function renderHourStrip() {{
  const strip = document.getElementById('hour-strip');
  strip.innerHTML = '';
  for (let h = 1; h <= 24; h++) {{
    const cell = document.createElement('div');
    cell.className = 'hour-cell';
    const label = document.createElement('div');
    label.className = 'hour-label';
    label.textContent = h;
    cell.appendChild(label);
    [['long', '\\u25b2'], ['flat', '\\u2013'], ['short', '\\u25bc']].forEach(([pos, symbol]) => {{
      const btn = document.createElement('button');
      btn.className = 'pos-btn' + (positions[h] === pos ? ' active ' + pos : '');
      btn.textContent = symbol;
      btn.onclick = () => {{ positions[h] = pos; renderHourStrip(); }};
      cell.appendChild(btn);
    }});
    strip.appendChild(cell);
  }}
}}

function reveal() {{
  if (!currentDate) return;
  const dam = SIM_DATA.dam[currentDate];
  const rtm = SIM_DATA.rtm[currentDate];
  if (!dam || !rtm) {{ document.getElementById('load-status').textContent = 'Missing real data for that date.'; return; }}

  let totalPnl = 0, optimalPnl = 0, correct = 0, directional = 0;
  const hours = [], spreads = [], colors = [];
  for (let h = 1; h <= 24; h++) {{
    const d = dam[h - 1], r = rtm[h - 1];
    const spread = (d != null && r != null) ? (d - r) : null;
    const pos = positions[h];
    let pnl = 0, isCorrect = null;
    if (spread != null) {{
      if (pos === 'long') {{ pnl = spread; directional++; isCorrect = spread > 0; }}
      else if (pos === 'short') {{ pnl = -spread; directional++; isCorrect = spread < 0; }}
      optimalPnl += Math.abs(spread);
      if (isCorrect) correct++;
    }}
    totalPnl += pnl;
    hours.push(h);
    spreads.push(spread);
    colors.push(pos === 'flat' ? '#666' : (isCorrect ? '#2ecc71' : '#e74c3c'));
  }}

  const pct = optimalPnl !== 0 ? (totalPnl / optimalPnl * 100) : (totalPnl === 0 ? 100 : 0);

  const statsEl = document.getElementById('result-stats');
  statsEl.innerHTML = '';
  const stats = [
    ['Your P&L', '$' + totalPnl.toFixed(1)],
    ['Optimal P&L (perfect hindsight)', '$' + optimalPnl.toFixed(1)],
    ['% of optimal', pct.toFixed(0) + '%'],
    ['Directional accuracy', directional > 0 ? (correct + '/' + directional) : 'n/a (all Flat)']
  ];
  stats.forEach(([label, value]) => {{
    const tile = document.createElement('div');
    tile.className = 'stat-tile';
    const l = document.createElement('div'); l.className = 'stat-label'; l.textContent = label;
    const v = document.createElement('div'); v.className = 'stat-value'; v.textContent = value;
    tile.appendChild(l); tile.appendChild(v);
    statsEl.appendChild(tile);
  }});

  Plotly.newPlot('result-chart', [{{
    x: hours, y: spreads, type: 'bar', marker: {{color: colors}},
    hovertemplate: 'Hour %{{x}}<br>Actual spread: $%{{y:.1f}}<extra></extra>'
  }}], {{
    template: 'plotly_dark', paper_bgcolor: '#161616', plot_bgcolor: '#161616',
    margin: {{t: 10, b: 40, l: 50, r: 20}},
    xaxis: {{title: 'Hour', dtick: 1, gridcolor: '#242424'}},
    yaxis: {{title: 'Actual spread (DAM - RTM, $/MWh)', gridcolor: '#242424', hoverformat: '.1f', zerolinecolor: '#666'}}
  }}, {{displayModeBar: false, responsive: true}});

  saveAttempt(currentDate, totalPnl, optimalPnl, pct, correct, directional);
  renderHistory();
  document.getElementById('result-section').classList.remove('hidden');
  document.getElementById('result-section').scrollIntoView({{behavior: 'smooth', block: 'start'}});
}}

const HISTORY_KEY = 'lrg_simulator_history';

function loadHistory() {{
  try {{ return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }} catch (e) {{ return []; }}
}}

function saveAttempt(date, totalPnl, optimalPnl, pct, correct, directional) {{
  const history = loadHistory().filter(a => a.date !== date);
  history.push({{date, totalPnl, optimalPnl, pct, correct, directional}});
  history.sort((a, b) => b.date.localeCompare(a.date));
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}}

function clearHistory() {{
  if (!confirm('Clear your entire attempt history? This cannot be undone.')) return;
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
}}

function renderHistory() {{
  const history = loadHistory();
  const table = document.getElementById('history-table');
  table.innerHTML = '';
  if (!history.length) {{
    table.innerHTML = '<tr><td style="color:#777; padding:8px 0; border-bottom:none;">No attempts saved yet.</td></tr>';
    return;
  }}
  const avgPct = history.reduce((s, a) => s + a.pct, 0) / history.length;
  const thead = document.createElement('thead');
  thead.innerHTML = '<tr><th>Date</th><th>Your P&amp;L</th><th>Optimal</th><th>% of optimal</th><th>Accuracy</th></tr>';
  const tbody = document.createElement('tbody');
  history.forEach(a => {{
    const row = document.createElement('tr');
    const cells = [
      a.date, '$' + a.totalPnl.toFixed(1), '$' + a.optimalPnl.toFixed(1),
      a.pct.toFixed(0) + '%', a.directional > 0 ? (a.correct + '/' + a.directional) : 'n/a'
    ];
    cells.forEach(text => {{ const td = document.createElement('td'); td.textContent = text; row.appendChild(td); }});
    tbody.appendChild(row);
  }});
  const tfoot = document.createElement('tfoot');
  const footRow = document.createElement('tr');
  ['Average (' + history.length + ' days)', '', '', avgPct.toFixed(0) + '%', ''].forEach(text => {{
    const td = document.createElement('td'); td.textContent = text; footRow.appendChild(td);
  }});
  tfoot.appendChild(footRow);
  table.appendChild(thead); table.appendChild(tbody); table.appendChild(tfoot);
}}

renderHistory();
</script>

</body>
</html>
"""

with open('docs/simulator.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Simulator generated successfully at docs/simulator.html!")
