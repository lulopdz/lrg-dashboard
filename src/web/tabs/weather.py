"""Weather Forecast tab HTML, stat tiles, stability table, and grid builders."""

import json

import pandas as pd

from dashboard_data import (
    COLORS,
    DAY_OPTIONS,
    MODEL_WEATHER_INPUTS,
    TABLE_DAYS,
    WEATHER_GROUPS,
    WEATHER_TILE_VARS,
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


def weather_tile_data():
    """{date_str: [{value, delta, direction} or None, ...]} for every day in the picker.

    Was baked once for tomorrow-vs-today while the charts below followed the Day selector, so
    moving the day left these showing another day's averages -- the same defect just fixed on
    the Supply Mix tab. Following the picker also generalises the comparison: the delta is now
    against the day before whichever day is selected, not against today specifically."""
    dates = weather['timestamp'].dt.date
    by_day = {d: weather[dates == d] for d in DAY_OPTIONS}
    out = {}
    for date in DAY_OPTIONS:
        day = by_day.get(date)
        if day is None or day.empty:
            continue
        prev = by_day.get(date - pd.Timedelta(days=1))
        entries = []
        for v in WEATHER_TILE_VARS:
            unit = WEATHER_VARS[v][1]
            avg = day[v].mean() if v in day.columns else None
            if avg is None or pd.isna(avg):
                entries.append(None)
                continue
            entry = {'value': f'{avg:.1f} {unit}'}
            prev_avg = prev[v].mean() if prev is not None and not prev.empty and v in prev.columns else None
            if prev_avg is not None and not pd.isna(prev_avg):
                delta = avg - prev_avg
                direction = 'up' if delta > 0.05 else ('down' if delta < -0.05 else 'flat')
                arrow = {'up': '▲', 'down': '▼', 'flat': '–'}[direction]
                entry['direction'] = direction
                entry['delta'] = f'{arrow} {delta:+.1f} {unit} vs previous day'
            entries.append(entry)
        if any(entries):
            out[str(date)] = entries
    return out


def weather_stat_tiles_html():
    """Empty shells; updateWeatherTiles() fills them from WEATHER_TILE_DATA for the selected
    day. The label is baked in since it never changes. Tile i is WEATHER_TILE_VARS[i], the same
    order weather_tile_data() writes its entries in."""
    return '\n'.join(
        f'<div class="stat-tile" id="weather-tile-{i}"><div class="stat-label">{weather_label_html(v)}</div>'
        f'<div class="stat-value"></div><div class="stat-delta"></div></div>'
        for i, v in enumerate(WEATHER_TILE_VARS)
    )


def weather_stability_html():
    """How much tomorrow's forecast has moved between successive model runs -- one column per
    lead time, so you can see whether it is converging or still wandering.

    Reading two lead times together is what makes this useful: if the 2-day-ago column is
    large and the 1-day-ago column is small, the forecast has settled and can be trusted. If
    both are large it is still moving. If the 1-day column is the larger of the two, something
    changed late, which is the least comfortable case.

    The ensemble spread (how far apart ECMWF's members are) is not repeated here -- it is
    already drawn as the shaded band on each chart above."""
    if weather_confidence is None:
        return ''
    tomorrow_date = today_date + pd.Timedelta(days=1)
    conf = weather_confidence[weather_confidence['timestamp'].dt.date == tomorrow_date]
    fc = weather[weather['timestamp'].dt.date == tomorrow_date]
    if conf.empty or fc.empty:
        return ''

    # Both figures are also shown as a share of how much the variable normally moves (its
    # hourly standard deviation over the trailing 30 days). The raw numbers can't be compared
    # across variables or judged on their own -- 67 W/m2 of solar spread sounds alarming until
    # you see solar's normal swing is 289, making it the *most* trustworthy row here. Dividing
    # by the mean instead would be wrong: temperature in Celsius is an interval scale (a "% of
    # 16 degrees" means nothing), and precipitation/snowfall average ~0 so it would explode.
    recent = weather[weather['timestamp'] >= weather['timestamp'].max() - pd.Timedelta(days=30)]

    def relative(value, var):
        sd = recent[var].std()
        if value is None or pd.isna(value) or not sd or sd < 1e-9:
            return ''
        return f' <span class="rel">({value / sd * 100:.0f}%)</span>'

    def cell(value, var, unit):
        if value is None or pd.isna(value):
            return '—'
        return f'{value:.1f} {unit}{relative(value, var)}'

    def revision(var, days_ago):
        """Mean absolute gap between the current forecast for tomorrow and what the run
        `days_ago` days earlier said for those same hours."""
        col = f'{var}_prev{days_ago}'
        if col not in conf.columns or not conf[col].notna().any():
            return None
        merged = fc[['hour', var]].merge(conf[['hour', col]], on='hour', how='inner').dropna()
        return (merged[var] - merged[col]).abs().mean() if not merged.empty else None

    rows = []
    for v in weather_var_keys:
        label, unit = weather_label_html(v), WEATHER_VARS[v][1]
        rev1, rev2 = revision(v, 1), revision(v, 2)
        if rev1 is None and rev2 is None:
            continue
        rows.append(f"""<tr><td>{label}</td>
  <td>{cell(rev1, v, unit)}</td>
  <td>{cell(rev2, v, unit)}</td></tr>""")
    if not rows:
        return ''

    return f"""
<h3>How much tomorrow's forecast has been moving</h3>
<table class="naive-table">
  <thead><tr><th>Variable</th><th>1 day ago</th><th>2 days ago</th></tr></thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>
<p class="caveat">Shift from what the model said 1 and 2 days ago for the same hours. Big 2-day next to
small 1-day = it has settled; both big = still wandering; 1-day bigger = something changed late.
Percentages are against each variable's normal hourly swing (30-day std), so rows compare.</p>"""


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


def build_weather_tab():
    """Build the Weather Forecast tab HTML, combo traces, and tile data JSON."""
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
    weather_tile_data_json = json.dumps(weather_tile_data())

    html = f"""<div id="tab-weather" class="tab-content">
<div class="summary-split">
  <div class="summary-half wide">
    <h3>Daily average for the selected day <span class="site">(price drivers)</span></h3>
    <div class="stat-row">
{weather_stat_tiles_html()}
    </div>
  </div>
  <div class="summary-half">
{weather_stability_html()}
  </div>
</div>
<div class="chart-legend">
  <span><i style="background:{COLORS['avg']}"></i>7d Average</span>
  <span><i style="background:{COLORS['prev_day']}"></i>Previous day</span>
  <span><i style="background:{COLORS['dam']}"></i>Selected day</span>
  <span><i class="band" style="background:rgba(52,152,219,0.35)"></i>Ensemble p10&ndash;p90</span>
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

    return html, combos, weather_tile_data_json
