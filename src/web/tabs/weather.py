"""Weather Forecast tab HTML: hour-by-hour context table, chart grid, and hourly table."""

import json

import pandas as pd

from dashboard_data import (
    COLORS,
    MODEL_WEATHER_INPUTS,
    TABLE_DAYS,
    WEATHER_GROUPS,
    WEATHER_VARS,
    default_forecast_date_idx,
    default_weather_idx,
    today_date,
    weather,
    weather_confidence,
    weather_label_html,
    weather_var_keys,
)
from dashboard_figures import (
    ENSEMBLE_TRACES,
    build_weather_grid_figs,
    build_wide_table_fig,
    compact_combos,
)
from tabs.common import WEATHER_LABELS_JSON, register_fig


def weather_grid_html(weather_grid_figs):
    """One small chart per weather variable (see build_weather_grid_figs) instead of one big
    chart with a variable dropdown -- every variable visible at once.

    Grouped by what the variable does to price (WEATHER_GROUPS). Sections mix sites, so each
    title names its site, and a dot marks the variables the forecast models actually read."""
    idx = {v: i for i, v in enumerate(weather_var_keys)}
    sections = []
    for heading, keys in WEATHER_GROUPS.items():
        tiles = []
        for v in keys:
            label, unit = WEATHER_VARS[v]
            div_id = f'weather-grid-{idx[v]}'
            mark = ' <span class="model-input" title="Forecast model input">●</span>' if v in MODEL_WEATHER_INPUTS else ''
            fig_html = weather_grid_figs[v].to_html(full_html=False, include_plotlyjs=False, div_id=div_id)
            tiles.append(f"""<div class="weather-tile">
  <h3>{weather_label_html(v)} ({unit}){mark}</h3>
  {fig_html}
  {register_fig(div_id, 3 + ENSEMBLE_TRACES, label, zones_json=json.dumps([label]), show_title=False)}
</div>""")
        sections.append(f"""<h3 class="grid-section">{heading}</h3>
<div class="weather-grid">
{chr(10).join(tiles)}
</div>""")
    return '\n'.join(sections)


CONTEXT_DAYS = 90        # history the table's shading and "unusual" marks are judged against
CONTEXT_UNUSUAL_Q = 0.8  # |value| in the top 20% of that span is flagged unusual

# Rows of the hour-by-hour table at the top of the tab: (key, label, site, unit, kind, help).
# kind: 'level' = shaded along its own 90-day range; 'change' = signed, flagged when unusual;
# 'spread' = shaded by size, flagged when unusual. Grouped like the charts below.
CONTEXT_ROWS = {
    "Demand": [
        ("temp", "Temperature", "Ottawa", "°C", "level", "Forecast temperature. The models read Ottawa's weather for every zone."),
        ("dT", "Change per hour", "Ottawa", "°C", "change", "Temperature change from the hour before: how steep the ramp is."),
        ("temp_tor", "Temperature", "Toronto", "°C", "level", "Toronto, the province's load centre. Context: no model reads it."),
    ],
    "Renewable supply": [
        ("wind", "Wind 100m", "Port Alma", "m/s", "level", "Wind at hub height where the wind fleet sits."),
        ("dW", "Change per hour", "Port Alma", "m/s", "change", "Wind change from the hour before: a wind ramp up or down."),
        ("solar", "Solar radiation", "Ottawa", "W/m²", "level", "Shortwave radiation; its evening drop starts the ramp."),
    ],
    "Forecast confidence": [
        ("rev", "Revision", "Ottawa temp", "°C", "change",
         "Temperature forecast 1 day before the hour minus 2 days before: how much it moved."),
        ("ensT", "Ensemble width", "Ottawa temp", "°C", "spread",
         "P10-P90 across ECMWF's 51 members: how divided the scenarios are."),
        ("ensW", "Ensemble width", "Port Alma wind", "m/s", "spread", "P10-P90 across ECMWF's 51 members for 100m wind."),
    ],
}
_SOURCE = {"temp": "temperature_2m", "temp_tor": "temperature_2m_toronto",
           "wind": "wind_speed_100m_port_alma", "solar": "shortwave_radiation"}


def weather_context_json():
    """Per day, HE1-24 values for every CONTEXT_ROWS row, plus what the page needs to shade and
    flag them: each level row's 5th-95th percentile range and each change/spread row's
    "unusual" |value| (top 20%), both over the last CONTEXT_DAYS days. renderWeatherContext()
    draws the table for whatever day the Day bar is on.

    Context, deliberately no trade/skip verdict: on 111 days (2026-07..11, three zones) steep
    ramps, forecast revisions and wide ensembles never made the picks worse -- picks in
    steep-ramp hours paid $11.4/h vs $7.5/h, wide-ensemble hours $5.0/h vs $2.3/h (TRACKING,
    2026-11-17). For past days the values are what the files hold now (near-actuals, a
    short-lead ensemble); weather_confidence_dayahead.csv keeps the day-before ensemble."""
    w = weather[['timestamp'] + [c for c in _SOURCE.values() if c in weather.columns]].sort_values('timestamp').copy()
    w = w.rename(columns={v: k for k, v in _SOURCE.items()})
    w['dT'] = w['temp'].diff()
    if 'wind' in w:
        w['dW'] = w['wind'].diff()
    w = w[w['timestamp'] >= w['timestamp'].max() - pd.Timedelta(days=CONTEXT_DAYS)].copy()
    w['date'] = w['timestamp'].dt.strftime('%Y-%m-%d')
    w['hour'] = w['timestamp'].dt.hour + 1
    if weather_confidence is not None:
        c = weather_confidence.copy()
        c['date'] = c['timestamp'].dt.strftime('%Y-%m-%d')
        if {'temperature_2m_prev1', 'temperature_2m_prev2'} <= set(c.columns):
            # Not "latest minus prev1": Open-Meteo's prev1 is the newest run at least 24h before
            # the hour, which for most of tomorrow IS the latest run, so that was always 0 there.
            c['rev'] = c['temperature_2m_prev1'] - c['temperature_2m_prev2']
        for key, var in (('ensT', 'temperature_2m'), ('ensW', 'wind_speed_100m_port_alma')):
            if {f'{var}_p10', f'{var}_p90'} <= set(c.columns):
                c[key] = c[f'{var}_p90'] - c[f'{var}_p10']
        extra = [k for k in ('rev', 'ensT', 'ensW') if k in c]
        w = w.merge(c[['date', 'hour'] + extra], on=['date', 'hour'], how='left')

    rows, scale, unusual = [], {}, {}
    for group, specs in CONTEXT_ROWS.items():
        for key, label, site, unit, kind, help_ in specs:
            if key not in w or not w[key].notna().any():
                continue
            rows.append({'key': key, 'group': group, 'label': label, 'site': site, 'unit': unit,
                         'kind': kind, 'help': help_})
            if kind == 'level':
                scale[key] = [round(float(w[key].quantile(q)), 1) for q in (0.05, 0.95)]
            else:
                unusual[key] = round(float(w[key].abs().quantile(CONTEXT_UNUSUAL_Q)), 2)
                scale[key] = [0, round(float(w[key].abs().quantile(0.95)), 2)]
    keys = [r['key'] for r in rows]
    days = {}
    for date, g in w.groupby('date'):
        g = g.set_index('hour').reindex(range(1, 25))
        days[date] = {k: [None if pd.isna(v) else round(float(v), 1) for v in g[k]] for k in keys}
    return json.dumps({'rows': rows, 'days': days, 'scale': scale, 'unusual': unusual})


def weather_context_html():
    """Shell for the hour-by-hour table; renderWeatherContext() fills it for the Day bar's day."""
    return f"""<div class="card wx-card">
  <div class="section-header">
    <h2 id="wx-title">Hour by hour</h2>
    <span class="wx-key"><span class="wx-swatch"></span>unusual (top 20% of the last {CONTEXT_DAYS} days)
      &nbsp; <span class="wx-star">&#9733;</span> the day's picks</span>
  </div>
  <div class="wx-scroll"><table class="wx-table" id="wx-table"></table></div>
  <p class="caveat wx-notes"><b>Forecast confidence</b> · <b>Revision</b>: forecast 1 day out minus 2 days out
  (0 = unchanged, &minus; = colder) · <b>Ensemble width</b>: P10&ndash;P90 of 51 scenarios now (near 0 = they agree).
  Big revision + narrow ensemble = changed its mind, now sure.<br>
  Context, not a skip signal: steep-ramp and wide-ensemble hours paid more ($11.4 vs $7.5/h, $5.0 vs $2.3/h; 111 days).</p>
</div>"""


def ensemble_age_html():
    """When tomorrow's ensemble band was fetched, for the legend. A failed daily fetch leaves the
    previous run's rows in place, which otherwise looks exactly like a fresh band; this says so.
    Empty for rows written before update_weather_confidence.py stamped fetched_at."""
    if weather_confidence is None or 'fetched_at' not in weather_confidence.columns:
        return ''
    tomorrow = weather_confidence[weather_confidence['timestamp'].dt.date == today_date + pd.Timedelta(days=1)]
    fetched = pd.to_datetime(tomorrow['fetched_at'], errors='coerce').dropna()
    if fetched.empty:
        return ''
    last = fetched.max()
    stale = last.date() < today_date
    label = f"fetched {last:%d/%m %H:%M}" + (" — not updated today" if stale else "")
    return f' <span class="site{" stale" if stale else ""}">({label})</span>'


def build_weather_tab():
    """Build the Weather Forecast tab HTML, combo traces, and the hour-by-hour table data (JSON)."""
    weather_grid_figs = build_weather_grid_figs(
        weather, 'timestamp', WEATHER_VARS,
        default_day_idx=default_forecast_date_idx,
        confidence=weather_confidence,
    )
    weather_table_fig = build_wide_table_fig(
        weather, 'timestamp', WEATHER_VARS, default_weather_idx, 'Weather', polished=True
    )

    combos = {
        f'weather-grid-{weather_var_keys.index(_v)}': compact_combos(_fig, 3 + ENSEMBLE_TRACES)
        for _v, _fig in weather_grid_figs.items()
    }

    html = f"""<div id="tab-weather" class="tab-content">
{weather_context_html()}
<div class="chart-legend">
  <span><i style="background:{COLORS['avg']}"></i>7d Average</span>
  <span><i style="background:{COLORS['prev_day']}"></i>Previous day</span>
  <span><i style="background:{COLORS['dam']}"></i>Selected day</span>
  <span><i class="band" style="background:rgba(52,152,219,0.35)"></i>Ensemble p10&ndash;p90{ensemble_age_html()}</span>
  <span><span class="model-input">●</span> Forecast model input</span>
</div>
{weather_grid_html(weather_grid_figs)}
<div class="card">
  <div class="section-header">
    <h2>Hourly Table (last {TABLE_DAYS} days)</h2>
    <button class="copy-btn" onclick="copyTableTSV(this, 'weather-table')">Copy</button>
  </div>
  {weather_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='weather-table')}
  <script>registerTable('weather-table', {WEATHER_LABELS_JSON});</script>
</div>
</div>"""

    return html, combos, weather_context_json()
