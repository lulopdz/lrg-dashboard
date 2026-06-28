import json
import math
import os

import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TABLE_BUCKET_SIZE = 100  # $/MWh step size for the discrete table color scales


def discrete_colorscale(zmin, zmax, palette, bucket_size=TABLE_BUCKET_SIZE):
    """Build a stepped (non-gradient) Plotly colorscale: one flat color per $bucket_size band."""
    n_buckets = max(1, math.ceil((zmax - zmin) / bucket_size))
    fractions = [i / max(n_buckets - 1, 1) for i in range(n_buckets)]
    colors = pcolors.sample_colorscale(palette, fractions)
    scale = []
    for i, color in enumerate(colors):
        scale.append([i / n_buckets, color])
        scale.append([(i + 1) / n_buckets, color])
    return scale


def discrete_diverging_colorscale(n_per_side=3, neg_palette='Reds', pos_palette='Greens'):
    """Strictly red (negative) / green (positive) -- no yellow midpoint -- n_per_side shades each."""
    shades = [i / (n_per_side - 1) * 0.6 + 0.3 for i in range(n_per_side)] if n_per_side > 1 else [0.6]
    neg_colors = list(reversed(pcolors.sample_colorscale(neg_palette, shades)))  # dark -> light, left to right
    pos_colors = pcolors.sample_colorscale(pos_palette, shades)  # light -> dark, left to right
    colors = neg_colors + pos_colors
    n_buckets = len(colors)
    scale = []
    for i, color in enumerate(colors):
        scale.append([i / n_buckets, color])
        scale.append([(i + 1) / n_buckets, color])
    return scale

TABLE_DAYS = 14
DEFAULT_ZONE = 'OTTAWA'

# GitHub repo that hosts this dashboard, used to build the links the "Refresh"
# buttons open (the GitHub Actions pages for each workflow).
GITHUB_OWNER = 'lulopdz'
GITHUB_REPO = 'lrg-dashboard'

# 1. Load DAM and RTM (both stored hourly; update_rtm.py aggregates the raw 5-min feed)
dam = pd.read_csv('data/ieso_dam_prices.csv', parse_dates=['interval_start_local'])
dam = dam.sort_values(['location', 'interval_start_local'])
dam['hour'] = dam['interval_start_local'].dt.hour + 1  # IESO hour-ending convention: 1-24

rtm = pd.read_csv('data/ieso_rtm_prices.csv', parse_dates=['interval_start_local'])
rtm = rtm.sort_values(['location', 'interval_start_local'])
rtm['hour'] = rtm['interval_start_local'].dt.hour + 1

weather = pd.read_csv('data/OTTAWA_weather.csv', parse_dates=['timestamp'])
weather = weather.sort_values('timestamp')
weather['hour'] = weather['timestamp'].dt.hour + 1

load_forecast = pd.read_csv('data/ieso_load_forecast.csv', parse_dates=['interval_start_local'])
load_forecast = load_forecast.sort_values('interval_start_local')
load_forecast['hour'] = load_forecast['interval_start_local'].dt.hour + 1

wind_forecast = pd.read_csv('data/ieso_wind_forecast.csv', parse_dates=['interval_start_local'])
wind_forecast = wind_forecast.sort_values(['zone', 'interval_start_local'])
wind_forecast['hour'] = wind_forecast['interval_start_local'].dt.hour + 1

zones = sorted(dam['location'].unique())
default_idx = zones.index(DEFAULT_ZONE)

WEATHER_VARS = {
    "temperature_2m": ("Temperature", "°C"),
    "wind_speed_10m": ("Wind Speed", "m/s"),
    "precipitation": ("Precipitation", "mm"),
    "snowfall": ("Snowfall", "cm"),
    "shortwave_radiation": ("Solar Radiation", "W/m²"),
    "relative_humidity_2m": ("Humidity", "%"),
}
weather_var_keys = list(WEATHER_VARS.keys())
default_weather_idx = 0  # Temperature

LOAD_VARS = {
    "ontario": ("Ontario (Total)", "MW"),
    "ontario_northeast": ("Northeast", "MW"),
    "ontario_northwest": ("Northwest", "MW"),
    "ontario_southwest": ("Southwest", "MW"),
    "ontario_southeast": ("Southeast", "MW"),
}
load_var_keys = list(LOAD_VARS.keys())
default_load_idx = 0  # Ontario (Total)

wind_zones = sorted(wind_forecast['zone'].unique())
default_wind_idx = wind_zones.index('Ontario Total') if 'Ontario Total' in wind_zones else 0

# 2. Spread = DAM - RTM, aligned by zone/hour
spread = dam[['location', 'interval_start_local', 'hour', 'lmp']].merge(
    rtm[['location', 'interval_start_local', 'hour', 'lmp']],
    on=['location', 'interval_start_local', 'hour'], suffixes=('_dam', '_rtm')
)
spread['lmp'] = spread['lmp_dam'] - spread['lmp_rtm']
spread = spread[['location', 'interval_start_local', 'hour', 'lmp']]

# 3. Shared reference date: the actual calendar day, in market time. DAM always publishes
# a day ahead (so its max date is "tomorrow", not "today"), and RTM is only ever as
# complete as "right now" -- neither dataset's max date is the right anchor. Using the
# real wall-clock date keeps DAM/RTM/Spread all defaulting to the same meaningful day.
today_date = pd.Timestamp.now(tz='UTC').tz_convert('-05:00').date()
table_start_date = today_date - pd.Timedelta(days=TABLE_DAYS - 1)
latest_ts = dam['interval_start_local'].max()
rtm_latest_ts = rtm['interval_start_local'].max()

# Dates used by the hourly tables: the rolling last TABLE_DAYS ending at today (unaffected by the Day selector)
SELECTABLE_DATES = [table_start_date + pd.Timedelta(days=d) for d in range(TABLE_DAYS)]
SELECTABLE_DATE_STRS = [str(d) for d in SELECTABLE_DATES]

# Dates selectable in the "Day" dropdown: same window, plus tomorrow (DAM is already
# published for it), defaulting to today rather than the last (tomorrow) entry.
DAY_OPTIONS = SELECTABLE_DATES + [today_date + pd.Timedelta(days=1)]
DAY_OPTION_STRS = [str(d) for d in DAY_OPTIONS]
default_date_idx = len(DAY_OPTIONS) - 2
default_weather_date_idx = len(DAY_OPTIONS) - 1  # tomorrow -- weather forecasts are most useful looking ahead


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


def build_hourly_fig(df, label, location_col='location', value_col='lmp', y_axis_title='Price ($/MWh)',
                      zones_list=None, default_zone_idx=None):
    """One zone-selector + the shared 'Day' selector both drive trace visibility via JS.
    location_col/value_col let this be reused for non-price datasets (e.g. Wind Forecast's
    'zone'/'generation_forecast') without duplicating the trace-building logic."""
    zones_list = zones_list if zones_list is not None else zones
    default_zone_idx = default_zone_idx if default_zone_idx is not None else default_idx

    fig = go.Figure()
    for zi, zone in enumerate(zones_list):
        df_zone = df[df[location_col] == zone]
        for di, date in enumerate(DAY_OPTIONS):
            visible = (zi == default_zone_idx and di == default_date_idx)
            prev_date = date - pd.Timedelta(days=1)
            day_z = df_zone[df_zone['interval_start_local'].dt.date == date].sort_values('hour')
            prev_z = df_zone[df_zone['interval_start_local'].dt.date == prev_date].sort_values('hour')
            week_start = date - pd.Timedelta(days=6)
            avg_window = df_zone[(df_zone['interval_start_local'].dt.date > week_start) & (df_zone['interval_start_local'].dt.date <= date)]
            avg_z = avg_window.groupby('hour')[value_col].mean().reset_index().sort_values('hour')

            fig.add_trace(go.Scatter(
                x=avg_z['hour'], y=avg_z[value_col], name='7d Average', mode='lines',
                line=dict(color='#888', dash='dot'), visible=visible, legendgroup=zone
            ))
            fig.add_trace(go.Scatter(
                x=prev_z['hour'], y=prev_z[value_col], name=str(prev_date), mode='lines+markers',
                line=dict(color='#f1c40f', dash='dash'), visible=visible, legendgroup=zone
            ))
            fig.add_trace(go.Scatter(
                x=day_z['hour'], y=day_z[value_col], name=str(date), mode='lines+markers',
                line=dict(color='#3498db', width=3), visible=visible, legendgroup=zone
            ))

    fig.update_layout(
        template='plotly_dark',
        title=f'{label} - Hourly Profile - {zones_list[default_zone_idx]} ({DAY_OPTION_STRS[default_date_idx]})',
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
        xaxis_title='Hour', yaxis_title=y_axis_title,
        xaxis=dict(dtick=1, range=[0.5, 24.5]),
        margin=dict(t=60, b=60, r=140),
        height=500
    )
    return fig


def build_spread_detail_fig():
    """Two stacked subplots sharing the hour axis: DAM vs RTM on top, spread sign bars below."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
        subplot_titles=('DAM vs RTM', 'Spread (DAM - RTM)')
    )
    for zi, zone in enumerate(zones):
        dam_zone = dam[dam['location'] == zone]
        rtm_zone = rtm[rtm['location'] == zone]
        for di, date in enumerate(DAY_OPTIONS):
            visible = (zi == default_idx and di == default_date_idx)
            dam_z = dam_zone[dam_zone['interval_start_local'].dt.date == date].sort_values('hour')
            rtm_z = rtm_zone[rtm_zone['interval_start_local'].dt.date == date].sort_values('hour')
            merged = dam_z[['hour', 'lmp']].merge(rtm_z[['hour', 'lmp']], on='hour', suffixes=('_dam', '_rtm'))
            merged['spread'] = merged['lmp_dam'] - merged['lmp_rtm']
            colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in merged['spread']]

            fig.add_trace(go.Scatter(
                x=dam_z['hour'], y=dam_z['lmp'], name='DAM', mode='lines+markers',
                line=dict(color='#3498db', width=2), visible=visible, legendgroup=zone
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=rtm_z['hour'], y=rtm_z['lmp'], name='RTM', mode='lines+markers',
                line=dict(color='#e67e22', width=2), visible=visible, legendgroup=zone
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=merged['hour'], y=merged['spread'], marker_color=colors,
                visible=visible, showlegend=False,
                hovertemplate='Hour %{x}<br>Spread: $%{y:.2f}<extra></extra>'
            ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        title=f'Spread (DAM - RTM) - {DEFAULT_ZONE} ({DAY_OPTION_STRS[default_date_idx]})',
        legend=dict(orientation='v', yanchor='middle', y=0.8, xanchor='left', x=1.02),
        margin=dict(t=60, b=40, r=140),
        height=650
    )
    fig.update_xaxes(dtick=1, range=[0.5, 24.5], row=1, col=1)
    fig.update_xaxes(dtick=1, range=[0.5, 24.5], title_text='Hour', row=2, col=1)
    fig.update_yaxes(title_text='Price ($/MWh)', row=1, col=1)
    fig.update_yaxes(title_text='Spread ($/MWh)', row=2, col=1)
    fig.add_hline(y=0, line_color='#666', line_width=1, row=2, col=1)
    return fig


def build_table_fig(df, label, diverging=False, palette='YlOrRd', location_col='location', value_col='lmp',
                     zones_list=None, default_zone_idx=None, colorbar_title='$/MWh',
                     hover_label='Price', hover_prefix='$', hover_suffix=''):
    """Unaffected by the Day selector: always the rolling last TABLE_DAYS ending at today_date."""
    zones_list = zones_list if zones_list is not None else zones
    default_zone_idx = default_zone_idx if default_zone_idx is not None else default_idx

    df_table = df[(df['interval_start_local'].dt.date >= table_start_date) & (df['interval_start_local'].dt.date <= today_date)].copy()
    df_table['date'] = df_table['interval_start_local'].dt.date.astype(str)

    # Fixed range shared across zones (instead of auto-scaling per zone) so the same
    # color always means the same price/spread, and intensity highlights extremes.
    if diverging:
        zmax = df_table[value_col].abs().max()
        zmin = -zmax
        colorscale = discrete_diverging_colorscale()
    else:
        zmin, zmax = 0, df_table[value_col].max()
        colorscale = discrete_colorscale(zmin, zmax, palette)
    heatmap_kwargs = dict(colorscale=colorscale, zmin=zmin, zmax=zmax)

    fig = go.Figure()
    for i, zone in enumerate(zones_list):
        pivot = (df_table[df_table[location_col] == zone]
                 .pivot_table(index='date', columns='hour', values=value_col, aggfunc='mean')
                 .reindex(index=SELECTABLE_DATE_STRS, columns=range(1, 25)))
        text = pivot.round(1).astype(str).values

        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            text=text, texttemplate='%{text}', textfont=dict(size=10),
            colorbar=dict(title=colorbar_title),
            visible=(i == default_zone_idx),
            hovertemplate=f'Date %{{y}}, Hour %{{x}}<br>{hover_label}: {hover_prefix}%{{z:.2f}}{hover_suffix}<extra></extra>',
            **heatmap_kwargs
        ))

    buttons = [
        dict(label=zone, method='update',
             args=[{'visible': [j == i for j in range(len(zones_list))]},
                   {'title': f'{label} - Hourly Table - {zone} (last {TABLE_DAYS} days)'}])
        for i, zone in enumerate(zones_list)
    ]
    fig.update_layout(
        template='plotly_dark',
        title=f'{label} - Hourly Table - {zones_list[default_zone_idx]} (last {TABLE_DAYS} days)',
        updatemenus=[dict(buttons=buttons, direction='down', x=1.0, y=1.12, xanchor='right', yanchor='top',
                           active=default_zone_idx, showactive=True)],
        xaxis_title='Hour', yaxis_title='Date',
        xaxis=dict(dtick=1, side='top'),
        yaxis=dict(tickmode='array', tickvals=SELECTABLE_DATE_STRS, ticktext=SELECTABLE_DATE_STRS),
        margin=dict(t=90, b=40, r=40),
        height=560
    )
    return fig


def build_wide_hourly_fig(df, time_col, var_map, default_var_idx, tab_label, default_day_idx=None):
    """Same 7d-avg / yesterday / today layout as build_hourly_fig, but looping over wide-format
    variables (columns, single location) instead of a 'location' column with one value column.
    Used by Weather (Open-Meteo columns) and Load Forecast (per-subzone columns)."""
    var_keys = list(var_map.keys())
    default_day_idx = default_day_idx if default_day_idx is not None else default_date_idx

    fig = go.Figure()
    for vi, var in enumerate(var_keys):
        for di, date in enumerate(DAY_OPTIONS):
            visible = (vi == default_var_idx and di == default_day_idx)
            prev_date = date - pd.Timedelta(days=1)
            day_z = df[df[time_col].dt.date == date].sort_values('hour')
            prev_z = df[df[time_col].dt.date == prev_date].sort_values('hour')
            week_start = date - pd.Timedelta(days=6)
            avg_window = df[(df[time_col].dt.date > week_start) & (df[time_col].dt.date <= date)]
            avg_z = avg_window.groupby('hour')[var].mean().reset_index().sort_values('hour')

            fig.add_trace(go.Scatter(
                x=avg_z['hour'], y=avg_z[var], name='7d Average', mode='lines',
                line=dict(color='#888', dash='dot'), visible=visible, legendgroup=var
            ))
            fig.add_trace(go.Scatter(
                x=prev_z['hour'], y=prev_z[var], name=str(prev_date), mode='lines+markers',
                line=dict(color='#f1c40f', dash='dash'), visible=visible, legendgroup=var
            ))
            fig.add_trace(go.Scatter(
                x=day_z['hour'], y=day_z[var], name=str(date), mode='lines+markers',
                line=dict(color='#3498db', width=3), visible=visible, legendgroup=var
            ))

    label0, unit0 = var_map[var_keys[default_var_idx]]
    fig.update_layout(
        template='plotly_dark',
        title=f'{tab_label} - Hourly Profile - {label0} ({DAY_OPTION_STRS[default_day_idx]})',
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
        xaxis_title='Hour', yaxis_title=f'{label0} ({unit0})',
        xaxis=dict(dtick=1, range=[0.5, 24.5]),
        margin=dict(t=60, b=60, r=140),
        height=500
    )
    return fig


def build_wide_table_fig(df, time_col, var_map, default_var_idx, tab_label, colorscale='Thermal'):
    """Same rolling-window heatmap as build_table_fig, with a variable dropdown (native Plotly
    updatemenu) instead of a zone dropdown, since each variable has its own scale/units."""
    var_keys = list(var_map.keys())
    df_table = df[(df[time_col].dt.date >= table_start_date) & (df[time_col].dt.date <= today_date)].copy()
    df_table['date'] = df_table[time_col].dt.date.astype(str)

    fig = go.Figure()
    for i, var in enumerate(var_keys):
        label, unit = var_map[var]
        pivot = (df_table.pivot_table(index='date', columns='hour', values=var, aggfunc='mean')
                 .reindex(index=SELECTABLE_DATE_STRS, columns=range(1, 25)))
        text = pivot.round(1).astype(str).values
        zmin, zmax = pivot.min().min(), pivot.max().max()
        if pd.isna(zmin) or pd.isna(zmax) or zmin == zmax:
            zmin, zmax = 0, 1

        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            text=text, texttemplate='%{text}', textfont=dict(size=10),
            colorscale=colorscale, zmin=zmin, zmax=zmax,
            colorbar=dict(title=unit),
            visible=(i == default_var_idx),
            hovertemplate=f'Date %{{y}}, Hour %{{x}}<br>{label}: %{{z:.1f}} {unit}<extra></extra>'
        ))

    buttons = [
        dict(label=var_map[var][0], method='update',
             args=[{'visible': [j == i for j in range(len(var_keys))]},
                   {'title': f'{tab_label} - Hourly Table - {var_map[var][0]} (last {TABLE_DAYS} days)'}])
        for i, var in enumerate(var_keys)
    ]
    label0 = var_map[var_keys[default_var_idx]][0]
    fig.update_layout(
        template='plotly_dark',
        title=f'{tab_label} - Hourly Table - {label0} (last {TABLE_DAYS} days)',
        updatemenus=[dict(buttons=buttons, direction='down', x=1.0, y=1.12, xanchor='right', yanchor='top',
                           active=default_var_idx, showactive=True)],
        xaxis_title='Hour', yaxis_title='Date',
        xaxis=dict(dtick=1, side='top'),
        yaxis=dict(tickmode='array', tickvals=SELECTABLE_DATE_STRS, ticktext=SELECTABLE_DATE_STRS),
        margin=dict(t=90, b=40, r=40),
        height=560
    )
    return fig


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

# 4. Assemble the HTML page with tabs (DAM / RTM / Spread / Weather / Load Forecast / Wind Forecast)
# and shared zone/day controls
os.makedirs('docs', exist_ok=True)

ZONES_JSON = json.dumps(zones)
DATES_JSON = json.dumps(DAY_OPTION_STRS)
WEATHER_LABELS_JSON = json.dumps([WEATHER_VARS[v][0] for v in weather_var_keys])
WEATHER_Y_AXIS_TITLES = [f'{WEATHER_VARS[v][0]} ({WEATHER_VARS[v][1]})' for v in weather_var_keys]
weather_latest_ts = weather['timestamp'].max()

LOAD_LABELS_JSON = json.dumps([LOAD_VARS[v][0] for v in load_var_keys])
LOAD_Y_AXIS_TITLES = [f'{LOAD_VARS[v][0]} ({LOAD_VARS[v][1]})' for v in load_var_keys]
load_latest_ts = load_forecast['interval_start_local'].max()

WIND_ZONES_JSON = json.dumps(wind_zones)
wind_latest_ts = wind_forecast['interval_start_local'].max()


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
</script>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('dam', this)">Day-Ahead Market</button>
  <button class="tab-btn" onclick="showTab('rtm', this)">Real-Time Market</button>
  <button class="tab-btn" onclick="showTab('spread', this)">Spread</button>
  <button class="tab-btn" onclick="showTab('weather', this)">Weather</button>
  <button class="tab-btn" onclick="showTab('load', this)">Load Forecast</button>
  <button class="tab-btn" onclick="showTab('wind', this)">Wind Forecast</button>
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
<h2>DAM - Hourly Table (last {TABLE_DAYS} days)</h2>
{dam_table_fig.to_html(full_html=False, include_plotlyjs=False)}
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
<h2>RTM - Hourly Table (last {TABLE_DAYS} days)</h2>
{rtm_table_fig.to_html(full_html=False, include_plotlyjs=False)}
</div>

<div id="tab-spread" class="tab-content">
<h2>Spread (DAM - RTM) - Hourly Profile (Positive = green, Negative = red)</h2>
{zone_control('spread-hourly')}
{spread_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='spread-hourly')}
{register_fig('spread-hourly', 3, 'Spread (DAM - RTM) - Hourly Profile')}
<h2>Spread (DAM - RTM) - Hourly Table (last {TABLE_DAYS} days)</h2>
{spread_table_fig.to_html(full_html=False, include_plotlyjs=False)}
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
<h2>Weather (OTTAWA) - Hourly Table (last {TABLE_DAYS} days)</h2>
{weather_table_fig.to_html(full_html=False, include_plotlyjs=False)}
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
<h2>Load Forecast - Hourly Table (last {TABLE_DAYS} days)</h2>
{load_table_fig.to_html(full_html=False, include_plotlyjs=False)}
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
<h2>Wind Forecast - Hourly Table (last {TABLE_DAYS} days)</h2>
{wind_table_fig.to_html(full_html=False, include_plotlyjs=False)}
</div>

<footer>DAM data through: {latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; RTM data through: {rtm_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Weather data through: {weather_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Load Forecast through: {load_latest_ts.strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; Wind Forecast through: {wind_latest_ts.strftime('%Y-%m-%d %H:%M')}</footer>

</body>
</html>
"""

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Dashboard generated successfully at docs/index.html!")
