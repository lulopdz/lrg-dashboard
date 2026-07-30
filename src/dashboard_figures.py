"""Plotly figure builders for the dashboard: hourly profile charts, the DAM/RTM/Spread
detail view, forecast charts, and the rolling hourly heatmap tables. Each function returns
a go.Figure; generar_web.py wires them into the page's tabs."""
import math

import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard_data import (
    DAY_OPTION_STRS, DAY_OPTIONS, DEFAULT_ZONE, SELECTABLE_DATE_STRS, TABLE_DAYS,
    dam, default_date_idx, default_idx, rtm, table_start_date, today_date, zones,
)

TABLE_BUCKET_SIZE = 100  # $/MWh step size for the discrete table color scales
TABLE_ROW_HEIGHT = 26    # px per date row in the hourly heatmap tables -- height scales with TABLE_DAYS instead of being fixed


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


def build_forecast_fig(forecast_df, meta, series_label='DAM'):
    """Tomorrow's predicted DAM/RTM/Spread curve for one zone, with the closest historical
    'similar day' (by forecasted load/wind/weather) plotted as a dashed reference, and the
    hour we're most confident in (lowest historical backtest error) marked on the axis."""
    is_spread = series_label == 'Spread'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['hour'], y=forecast_df['analog_lmp'],
        name=f"Similar day ({meta.get('analog_date')})", mode='lines+markers',
        line=dict(color='#888', dash='dash')
    ))
    if 'analog_lmp_2' in forecast_df.columns and forecast_df['analog_lmp_2'].notna().any():
        fig.add_trace(go.Scatter(
            x=forecast_df['hour'], y=forecast_df['analog_lmp_2'],
            name=f"2nd similar day ({meta.get('analog_date_2')})", mode='lines+markers',
            line=dict(color='#666', dash='dot')
        ))
    fig.add_trace(go.Scatter(
        x=forecast_df['hour'], y=forecast_df['predicted_lmp'],
        name=f"Predicted ({meta.get('target_date')})", mode='lines+markers',
        line=dict(color='#9b59b6', width=3)
    ))
    if is_spread:
        fig.add_hline(y=0, line_color='#666', line_width=1)

    recommended = meta.get('recommended_hour')
    if recommended and recommended.get('hour'):
        fig.add_vline(
            x=recommended['hour'], line_color='#2ecc71', line_width=1, line_dash='dot',
            annotation_text=f"Most confident: hour {recommended['hour']}",
            annotation_position='top', annotation_font_color='#2ecc71'
        )

    fig.update_layout(
        template='plotly_dark',
        title=f"{series_label} Forecast - {meta.get('zone')} ({meta.get('target_date')})",
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
        xaxis_title='Hour', yaxis_title=f"{'Spread' if is_spread else 'Price'} ($/MWh)",
        xaxis=dict(dtick=1, range=[0.5, 24.5]),
        margin=dict(t=60, b=60, r=140),
        height=500
    )
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
            text=text, texttemplate='%{text}', textfont=dict(size=11),
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
        height=130 + TABLE_DAYS * TABLE_ROW_HEIGHT
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
            text=text, texttemplate='%{text}', textfont=dict(size=11),
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
        height=130 + TABLE_DAYS * TABLE_ROW_HEIGHT
    )
    return fig
