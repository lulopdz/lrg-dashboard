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
from theme import COLORS, PROFILE_HEIGHT, SPREAD_HEIGHT, TABLE_BUCKET_SIZE, TABLE_ROW_HEIGHT


def hour_xaxis(**extra):
    base = dict(dtick=1, range=[0.5, 24.5])
    base.update(extra)
    return base


def _reference_series(df, dates, value_col, date):
    """The three reference series every 'hourly profile' chart plots for a given day: the day
    itself, the day before it, and the trailing 7-day average leading up to it (all sorted by
    hour). Shared by build_hourly_fig, build_wide_hourly_fig, and build_weather_grid_figs,
    which otherwise differ too much in trace styling (marker size, fill, legend grouping) to
    also unify into one function.

    `dates` is df's timestamp column already reduced to plain dates (pass df[col].dt.date).
    It's a parameter rather than computed here because every caller loops this over all 31
    DAY_OPTIONS with the same df: .dt.date materializes a full object-dtype column, and doing
    it 4x per call inside those loops was the single most expensive thing in the page build."""
    prev_date = date - pd.Timedelta(days=1)
    day_z = df[dates == date].sort_values('hour')
    prev_z = df[dates == prev_date].sort_values('hour')
    week_start = date - pd.Timedelta(days=6)
    avg_window = df[(dates > week_start) & (dates <= date)]
    avg_z = avg_window.groupby('hour')[value_col].mean().reset_index().sort_values('hour')
    return day_z, prev_z, avg_z, prev_date


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
                      zones_list=None, default_zone_idx=None, polished=False, default_day_idx=None,
                      compact_zones=False):
    """One zone-selector + the shared 'Day' selector both drive trace visibility via JS.
    location_col/value_col let this be reused for non-price datasets (e.g. Wind Forecast's
    'zone'/'generation_forecast') without duplicating the trace-building logic.
    polished=True opts into the refreshed mark/hover treatment (bigger ringed markers, a
    soft fill under the 'Today' line, unified crosshair hover) -- kept opt-in so tabs can
    pick it up one at a time instead of every build_hourly_fig call changing at once.
    default_day_idx overrides which DAY_OPTIONS entry starts visible (Wind Forecast opens on
    tomorrow; everything else defaults to today).
    compact_zones=True only pre-bakes the default zone's traces (still one hidden trace set
    per day, toggled instantly) and appends 3 empty 'dynamic' placeholder traces that the
    page's JS fills in on demand -- via zone_hourly_data()'s compact {zone: {date: [24]}} data
    -- when a non-default zone is picked, instead of pre-baking every zone x day combination.
    Cuts this figure's embedded data by roughly (zones - 1) / zones."""
    zones_list = zones_list if zones_list is not None else zones
    default_zone_idx = default_zone_idx if default_zone_idx is not None else default_idx
    default_day_idx = default_day_idx if default_day_idx is not None else default_date_idx

    baked_zones = [zones_list[default_zone_idx]] if compact_zones else zones_list

    fig = go.Figure()
    for zi, zone in enumerate(baked_zones):
        df_zone = df[df[location_col] == zone]
        zone_dates = df_zone['interval_start_local'].dt.date
        for di, date in enumerate(DAY_OPTIONS):
            visible = di == default_day_idx if compact_zones else (zi == default_zone_idx and di == default_day_idx)
            day_z, prev_z, avg_z, prev_date = _reference_series(df_zone, zone_dates, value_col, date)

            fig.add_trace(go.Scatter(
                x=avg_z['hour'], y=avg_z[value_col], name='7d Average', mode='lines',
                line=dict(color=COLORS['avg'] if polished else COLORS['avg_legacy'], dash='dot', width=1.5 if polished else 2),
                visible=visible, legendgroup=zone
            ))
            fig.add_trace(go.Scatter(
                x=prev_z['hour'], y=prev_z[value_col], name=str(prev_date), mode='lines+markers',
                line=dict(color=COLORS['prev_day'] if polished else COLORS['prev_day_legacy'], dash='dash'),
                marker=dict(size=7, line=dict(width=1.5, color=COLORS['ring'])) if polished else {},
                visible=visible, legendgroup=zone
            ))
            fig.add_trace(go.Scatter(
                x=day_z['hour'], y=day_z[value_col], name=str(date), mode='lines+markers',
                line=dict(color=COLORS['dam'], width=3),
                marker=dict(size=8, line=dict(width=2, color=COLORS['ring'])) if polished else {},
                fill='tozeroy' if polished else None, fillcolor='rgba(52,152,219,0.08)' if polished else None,
                visible=visible, legendgroup=zone
            ))

    if compact_zones:
        # Same 3 mark specs as above (avg/prev/today), empty and hidden until the page's JS
        # (applyFigSelection -> showCompactZone) restyles their x/y/name for whichever
        # non-default zone is currently selected.
        fig.add_trace(go.Scatter(
            x=[], y=[], name='7d Average', mode='lines',
            line=dict(color=COLORS['avg'] if polished else COLORS['avg_legacy'], dash='dot', width=1.5 if polished else 2),
            visible=False
        ))
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='lines+markers',
            line=dict(color=COLORS['prev_day'] if polished else COLORS['prev_day_legacy'], dash='dash'),
            marker=dict(size=7, line=dict(width=1.5, color=COLORS['ring'])) if polished else {},
            visible=False
        ))
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='lines+markers',
            line=dict(color=COLORS['dam'], width=3),
            marker=dict(size=8, line=dict(width=2, color=COLORS['ring'])) if polished else {},
            fill='tozeroy' if polished else None, fillcolor='rgba(52,152,219,0.08)' if polished else None,
            visible=False
        ))

    xaxis = hour_xaxis()
    if polished:
        xaxis.update(showspikes=True, spikemode='across', spikesnap='cursor',
                      spikedash='dot', spikethickness=1, spikecolor=COLORS['muted'],
                      gridcolor=COLORS['grid'])

    # polished tabs drop the in-canvas title: it only repeated the section heading and the
    # zone/day selectors above the chart, and removing it reclaims top margin for the plot.
    title = None if polished else f'{label} - Hourly Profile - {zones_list[default_zone_idx]} ({DAY_OPTION_STRS[default_day_idx]})'

    yaxis = dict(gridcolor=COLORS['grid'], hoverformat='.1f') if polished else dict(hoverformat='.1f')

    fig.update_layout(
        template='plotly_dark',
        title=title,
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
        xaxis_title='Hour', yaxis_title=y_axis_title,
        xaxis=xaxis,
        yaxis=yaxis,
        hovermode='x unified' if polished else 'closest',
        margin=dict(t=30, b=60, r=140) if polished else dict(t=60, b=60, r=140),
        height=PROFILE_HEIGHT if polished else 500
    )
    return fig


def build_spread_detail_fig(polished=False, compact_zones=False):
    """Two stacked subplots sharing the hour axis: DAM vs RTM on top, spread sign bars below.
    polished=True applies the same treatment as build_hourly_fig(polished=True): drops the
    redundant in-canvas title, adds ringed markers, and turns on unified crosshair hover.
    The two subplot_titles ('DAM vs RTM' / 'Spread...') stay either way -- unlike the main
    title, they label two different panels and aren't repeated anywhere else on the page.
    compact_zones=True mirrors build_hourly_fig's compact_zones: only the default zone's
    day-by-day traces are pre-baked (still one hidden 3-trace set per day, toggled
    instantly), plus 3 empty 'dynamic' placeholder traces -- one per row/col position a real
    combo occupies -- that the page's JS fills in for any other zone. Unlike build_hourly_fig,
    Spread doesn't need its own copy of the compact {zone: {date: [24]}} data: it's just
    DAM - RTM, both of which zone_hourly_data() in generar_web.py already embeds for the
    DAM/RTM hourly tabs, so the JS derives the spread client-side instead of shipping a
    third copy of the same numbers."""
    baked_zones = [DEFAULT_ZONE] if compact_zones else zones
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
        subplot_titles=('DAM vs RTM', 'Spread (DAM - RTM)')
    )
    marker = dict(size=7, line=dict(width=1.5, color=COLORS['ring'])) if polished else {}
    for zi, zone in enumerate(baked_zones):
        dam_zone = dam[dam['location'] == zone]
        rtm_zone = rtm[rtm['location'] == zone]
        for di, date in enumerate(DAY_OPTIONS):
            visible = di == default_date_idx if compact_zones else (zi == default_idx and di == default_date_idx)
            dam_z = dam_zone[dam_zone['interval_start_local'].dt.date == date].sort_values('hour')
            rtm_z = rtm_zone[rtm_zone['interval_start_local'].dt.date == date].sort_values('hour')
            merged = dam_z[['hour', 'lmp']].merge(rtm_z[['hour', 'lmp']], on='hour', suffixes=('_dam', '_rtm'))
            merged['spread'] = merged['lmp_dam'] - merged['lmp_rtm']
            colors = [COLORS['positive'] if v >= 0 else COLORS['negative'] for v in merged['spread']]

            fig.add_trace(go.Scatter(
                x=dam_z['hour'], y=dam_z['lmp'], name='DAM', mode='lines+markers',
                line=dict(color=COLORS['dam'], width=2), marker=marker, visible=visible, legendgroup=zone
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=rtm_z['hour'], y=rtm_z['lmp'], name='RTM', mode='lines+markers',
                line=dict(color=COLORS['rtm'], width=2), marker=marker, visible=visible, legendgroup=zone
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=merged['hour'], y=merged['spread'], marker_color=colors,
                visible=visible, showlegend=False,
                hovertemplate='Hour %{x}<br>Spread: $%{y:.1f}<extra></extra>'
            ), row=2, col=1)

    if compact_zones:
        fig.add_trace(go.Scatter(
            x=[], y=[], name='DAM', mode='lines+markers',
            line=dict(color=COLORS['dam'], width=2), marker=marker, visible=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[], y=[], name='RTM', mode='lines+markers',
            line=dict(color=COLORS['rtm'], width=2), marker=marker, visible=False
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[], y=[], marker_color=[], visible=False, showlegend=False,
            hovertemplate='Hour %{x}<br>Spread: $%{y:.1f}<extra></extra>'
        ), row=2, col=1)

    title = None if polished else f'Spread (DAM - RTM) - {DEFAULT_ZONE} ({DAY_OPTION_STRS[default_date_idx]})'
    fig.update_layout(
        template='plotly_dark',
        title=title,
        legend=dict(orientation='v', yanchor='middle', y=0.8, xanchor='left', x=1.02),
        hovermode='x unified' if polished else 'closest',
        margin=dict(t=30, b=40, r=140) if polished else dict(t=60, b=40, r=140),
        height=SPREAD_HEIGHT if polished else 650
    )
    row_xaxis = hour_xaxis()
    if polished:
        row_xaxis.update(showspikes=True, spikemode='across', spikesnap='cursor',
                          spikedash='dot', spikethickness=1, spikecolor=COLORS['muted'], gridcolor=COLORS['grid'])
    fig.update_xaxes(row=1, col=1, **row_xaxis)
    fig.update_xaxes(title_text='Hour', row=2, col=1, **row_xaxis)

    row1_yaxis = dict(title_text='Price ($/MWh)', hoverformat='.1f')
    row2_yaxis = dict(title_text='Spread ($/MWh)', hoverformat='.1f')
    if polished:
        row1_yaxis['gridcolor'] = row2_yaxis['gridcolor'] = COLORS['grid']
    fig.update_yaxes(row=1, col=1, **row1_yaxis)
    fig.update_yaxes(row=2, col=1, **row2_yaxis)
    fig.add_hline(y=0, line_color=COLORS['muted'], line_width=1, row=2, col=1)
    return fig


def build_forecast_fig(forecast_df, meta, series_label='DAM'):
    """Tomorrow's predicted DAM/RTM/Spread curve for one zone, with the closest historical
    'similar day' (by forecasted load/wind/weather) plotted as a dashed reference, and the
    hour we're most confident in (lowest historical backtest error) marked on the axis.
    Same ringed-marker/crosshair/unified-hover/no-title treatment as the rest of the site
    (see build_hourly_fig's polished=True) -- the outer <h2> in the forecast tab already
    carries the title."""
    is_spread = series_label == 'Spread'
    ring_marker = dict(size=7, line=dict(width=1.5, color=COLORS['ring']))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['hour'], y=forecast_df['analog_lmp'],
        name=f"Similar #1 ({meta.get('analog_date')})", mode='lines+markers',
        line=dict(color=COLORS['avg_legacy'], dash='dash', width=2.5), marker=ring_marker
    ))
    if 'analog_lmp_2' in forecast_df.columns and forecast_df['analog_lmp_2'].notna().any():
        fig.add_trace(go.Scatter(
            x=forecast_df['hour'], y=forecast_df['analog_lmp_2'],
            name=f"Similar #2 ({meta.get('analog_date_2')})", mode='lines+markers',
            line=dict(color=COLORS['muted'], dash='dot', width=2.5), marker=ring_marker
        ))

    # Confidence band: predicted +/- the model's historical per-hour MAE from the backtest
    # (see forecast_common.backtest's hourly_mae) -- a "typical error range" for that hour,
    # not a calibrated statistical interval (that would need quantile regression), and
    # symmetric even though real price errors likely skew toward spikes. Simple first pass
    # using data already computed for the "most confident hour" stat.
    hourly_mae = (meta.get('backtest') or {}).get('hourly_mae') or {}
    band_err = forecast_df['hour'].map(lambda h: hourly_mae.get(str(h))).astype(float)
    if band_err.notna().any():
        fig.add_trace(go.Scatter(
            x=forecast_df['hour'], y=forecast_df['predicted_lmp'] - band_err, mode='lines',
            line=dict(width=0), hoverinfo='skip', showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['hour'], y=forecast_df['predicted_lmp'] + band_err, mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor='rgba(155,89,182,0.18)',
            hoverinfo='skip', name='Error range (±MAE)'
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df['hour'], y=forecast_df['predicted_lmp'],
        name='Predicted', mode='lines+markers',
        line=dict(color=COLORS['predicted'], width=3), marker=dict(size=8, line=dict(width=2, color=COLORS['ring']))
    ))
    if is_spread:
        fig.add_hline(y=0, line_color=COLORS['muted'], line_width=1)

    recommended = meta.get('recommended_hour')
    if recommended and recommended.get('hour'):
        fig.add_vline(
            x=recommended['hour'], line_color=COLORS['positive'], line_width=1, line_dash='dot',
            annotation_text=f"Most confident: hour {recommended['hour']}",
            annotation_position='top', annotation_font_color=COLORS['positive']
        )

    fig.update_layout(
        template='plotly_dark',
        title=None,
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
        xaxis_title='Hour', yaxis_title=f"{'Spread' if is_spread else 'Price'} ($/MWh)",
        xaxis=hour_xaxis(showspikes=True, spikemode='across', spikesnap='cursor',
                          spikedash='dot', spikethickness=1, spikecolor=COLORS['muted'], gridcolor=COLORS['grid']),
        yaxis=dict(gridcolor=COLORS['grid'], hoverformat='.1f'),
        hovermode='x unified',
        margin=dict(t=30, b=60, r=140),
        height=PROFILE_HEIGHT
    )
    return fig


def build_analog_comparison_fig(meta):
    """'Why this day?' as a dumbbell chart instead of a wide table: one row per analog-search
    variable, each plotted as tomorrow's forecast expressed as a % difference from that
    analog day's actual value (0% = exact match), with a stem back to the zero line. % is the
    one unit every variable (MW, degC, mm, W/m2...) can share on one axis without a table's
    column-per-unit sprawl. Marker size carries ANALOG_FEATURE_WEIGHTS -- the variables that
    actually drove the similarity search stand out -- and rows are sorted the same way, most
    heavily-weighted at top. A variable whose analog value is exactly 0 (e.g. no rain that
    day) has no defined % difference and is skipped for that day, same as the table it
    replaces did via 'n/a'. Returns None if there's nothing to compare (no analog found)."""
    rows = meta.get('analog_comparison') or []
    if not rows:
        return None
    rows_2 = meta.get('analog_comparison_2') or []
    order = sorted(range(len(rows)), key=lambda i: -rows[i].get('weight', 1))
    AXIS_LIMIT = 50  # a variable that swings further than this (e.g. rain forecast going to/from
    # near-zero, which blows up a % difference) gets pinned to the edge instead of stretching
    # the whole chart's scale to fit one outlier -- the arrow marker + hover note say so.

    def pct(row):
        return None if row['analog'] == 0 else (row['target'] - row['analog']) / row['analog'] * 100

    def series(rows_ordered):
        # Literal unicode (·, ±), not &middot;/&plusmn; -- Plotly hover text only understands
        # a small fixed set of real HTML tags (<br>, <b>, <i>, ...), not named entities, so
        # those would print as literal "&middot;" in the popup instead of being decoded.
        stem_x, stem_y = [], []
        dot_x, dot_y, dot_size, dot_symbol, hover = [], [], [], [], []
        for i in order:
            r = rows_ordered[i]
            p = pct(r)
            if p is None:
                continue
            clamped = max(-AXIS_LIMIT, min(AXIS_LIMIT, p))
            overflow = clamped != p
            stem_x += [0, clamped, None]
            stem_y += [r['label'], r['label'], None]
            dot_x.append(clamped)
            dot_y.append(r['label'])
            dot_size.append(8 + 6 * (r.get('weight', 1) - 1))
            dot_symbol.append(('triangle-right' if p > 0 else 'triangle-left') if overflow else 'circle')
            note = f" (chart capped at ±{AXIS_LIMIT}%)" if overflow else ""
            hover.append(f"{r['target']:.1f} {r['unit']} vs {r['analog']:.1f} {r['unit']} · weight {r.get('weight', 1):g}x"
                         f"<br>{p:+.0f}% vs analog{note}")
        return stem_x, stem_y, dot_x, dot_y, dot_size, dot_symbol, hover

    fig = go.Figure()
    fig.add_vline(x=0, line_color=COLORS['muted'], line_width=1)

    stem_x, stem_y, dot_x, dot_y, dot_size, dot_symbol, hover = series(rows)
    fig.add_trace(go.Scatter(x=stem_x, y=stem_y, mode='lines', line=dict(color=COLORS['muted'], width=1.5),
                              hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(
        x=dot_x, y=dot_y, mode='markers', name=f"Similar #1 ({meta.get('analog_date')})",
        marker=dict(size=dot_size, symbol=dot_symbol, color=COLORS['avg_legacy'], line=dict(width=1.5, color=COLORS['ring'])),
        customdata=hover, hovertemplate='%{y}<br>%{customdata}<extra></extra>'
    ))

    if rows_2:
        stem_x2, stem_y2, dot_x2, dot_y2, dot_size2, dot_symbol2, hover2 = series(rows_2)
        symbol2 = [(s + '-open' if s != 'circle' else 'diamond') for s in dot_symbol2]
        fig.add_trace(go.Scatter(x=stem_x2, y=stem_y2, mode='lines', line=dict(color=COLORS['muted'], width=1, dash='dot'),
                                  hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(
            x=dot_x2, y=dot_y2, mode='markers', name=f"Similar #2 ({meta.get('analog_date_2')})",
            marker=dict(size=[max(4, s - 3) for s in dot_size2], symbol=symbol2, color=COLORS['muted'],
                        line=dict(width=1.5, color=COLORS['ring'])),
            customdata=hover2, hovertemplate='%{y}<br>%{customdata}<extra></extra>'
        ))

    fig.update_layout(
        template='plotly_dark', title=None,
        xaxis=dict(title="Tomorrow's forecast vs. analog day", ticksuffix='%', gridcolor=COLORS['grid'], zeroline=False,
                    range=[-AXIS_LIMIT, AXIS_LIMIT]),
        yaxis=dict(autorange='reversed', gridcolor=COLORS['grid']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(t=40, b=40, l=190, r=30),
        height=60 + len(rows) * 34,
    )
    return fig


def build_table_fig(df, label, diverging=False, palette='YlOrRd', location_col='location', value_col='lmp',
                     zones_list=None, default_zone_idx=None, colorbar_title='$/MWh',
                     hover_label='Price', hover_prefix='$', hover_suffix='', polished=False):
    """Unaffected by the Day selector: always the rolling last TABLE_DAYS ending at today_date.
    polished=True drops the in-canvas title -- the zone dropdown already names the zone and
    the card heading above already says 'Hourly Table', so the title only repeated both."""
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
            # Plain Python lists, not the numpy/pandas objects themselves: plotly.py encodes
            # numpy-backed numeric arrays as compact base64 ({"dtype":...,"bdata":...}) which
            # copyTableTSV (see generar_web.py) can't read directly off the rendered figure --
            # plain lists always serialize as ordinary JSON arrays.
            z=pivot.values.tolist(), x=list(pivot.columns), y=list(pivot.index),
            text=text, texttemplate='%{text}', textfont=dict(size=11),
            colorbar=dict(title=colorbar_title),
            visible=(i == default_zone_idx),
            hovertemplate=f'Date %{{y}}, Hour %{{x}}<br>{hover_label}: {hover_prefix}%{{z:.1f}}{hover_suffix}<extra></extra>',
            **heatmap_kwargs
        ))

    title = None if polished else f'{label} - Hourly Table - {zones_list[default_zone_idx]} (last {TABLE_DAYS} days)'
    layout_kwargs = dict(
        template='plotly_dark',
        title=title,
        xaxis_title='Hour', yaxis_title='Date',
        xaxis=dict(dtick=1, side='top'),
        yaxis=dict(tickmode='array', tickvals=SELECTABLE_DATE_STRS, ticktext=SELECTABLE_DATE_STRS),
        margin=dict(t=50, b=40, r=40) if polished else dict(t=90, b=40, r=40),
        height=(100 if polished else 130) + TABLE_DAYS * TABLE_ROW_HEIGHT
    )
    if polished:
        # Zone switching is driven externally (the shared day-bar selector) instead of
        # Plotly's own dropdown, so there's nothing left to wire buttons/updatemenus to.
        pass
    else:
        buttons = [
            dict(label=zone, method='update',
                 args=[{'visible': [j == i for j in range(len(zones_list))]},
                       {'title': f'{label} - Hourly Table - {zone} (last {TABLE_DAYS} days)'}])
            for i, zone in enumerate(zones_list)
        ]
        layout_kwargs['updatemenus'] = [dict(buttons=buttons, direction='down', x=1.0, y=1.12, xanchor='right', yanchor='top',
                                              active=default_zone_idx, showactive=True)]
    fig.update_layout(**layout_kwargs
    )
    return fig


def build_wide_hourly_fig(df, time_col, var_map, default_var_idx, tab_label, default_day_idx=None, polished=False):
    """Same 7d-avg / yesterday / today layout as build_hourly_fig, but looping over wide-format
    variables (columns, single location) instead of a 'location' column with one value column.
    Used by Weather (Open-Meteo columns) and Load Forecast (per-subzone columns).
    polished=True applies the same ringed-marker/crosshair/no-title treatment as
    build_hourly_fig(polished=True) -- minus the tozeroy fill, since zero isn't a meaningful
    baseline for most weather variables (temperature goes negative, humidity/wind don't
    have a $-style floor) the way it is for price."""
    var_keys = list(var_map.keys())
    default_day_idx = default_day_idx if default_day_idx is not None else default_date_idx

    df_dates = df[time_col].dt.date
    fig = go.Figure()
    for vi, var in enumerate(var_keys):
        for di, date in enumerate(DAY_OPTIONS):
            visible = (vi == default_var_idx and di == default_day_idx)
            day_z, prev_z, avg_z, prev_date = _reference_series(df, df_dates, var, date)

            fig.add_trace(go.Scatter(
                x=avg_z['hour'], y=avg_z[var], name='7d Average', mode='lines',
                line=dict(color=COLORS['avg'] if polished else COLORS['avg_legacy'], dash='dot', width=1.5 if polished else 2),
                visible=visible, legendgroup=var
            ))
            fig.add_trace(go.Scatter(
                x=prev_z['hour'], y=prev_z[var], name=str(prev_date), mode='lines+markers',
                line=dict(color=COLORS['prev_day'] if polished else COLORS['prev_day_legacy'], dash='dash'),
                marker=dict(size=7, line=dict(width=1.5, color=COLORS['ring'])) if polished else {},
                visible=visible, legendgroup=var
            ))
            fig.add_trace(go.Scatter(
                x=day_z['hour'], y=day_z[var], name=str(date), mode='lines+markers',
                line=dict(color=COLORS['dam'], width=3),
                marker=dict(size=8, line=dict(width=2, color=COLORS['ring'])) if polished else {},
                visible=visible, legendgroup=var
            ))

    label0, unit0 = var_map[var_keys[default_var_idx]]
    xaxis = hour_xaxis()
    if polished:
        xaxis.update(showspikes=True, spikemode='across', spikesnap='cursor',
                      spikedash='dot', spikethickness=1, spikecolor=COLORS['muted'], gridcolor=COLORS['grid'])

    title = None if polished else f'{tab_label} - Hourly Profile - {label0} ({DAY_OPTION_STRS[default_day_idx]})'
    yaxis = dict(gridcolor=COLORS['grid'], hoverformat='.1f') if polished else dict(hoverformat='.1f')

    fig.update_layout(
        template='plotly_dark',
        title=title,
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
        xaxis_title='Hour', yaxis_title=f'{label0} ({unit0})',
        xaxis=xaxis,
        yaxis=yaxis,
        hovermode='x unified' if polished else 'closest',
        margin=dict(t=30, b=60, r=140) if polished else dict(t=60, b=60, r=140),
        height=PROFILE_HEIGHT if polished else 500
    )
    return fig


ENSEMBLE_TRACES = 2  # the p10/p90 pair build_weather_grid_figs prepends to each day's group


def build_weather_grid_figs(df, time_col, var_map, default_day_idx=None, confidence=None):
    """Small multiples: one compact chart per variable instead of one big chart with a
    variable dropdown, so every variable (temperature, wind, precipitation...) is visible
    at a glance. Same 7d-avg / previous-day / selected-day trace layout as
    build_wide_hourly_fig, just split one-figure-per-variable; the shared 3-color meaning
    is explained once via an external legend instead of repeating a legend on every tile.
    Each figure is registered with a single-item 'zones' list (itself) purely so it can
    reuse the existing registerFig/applyFigSelection date-sync machinery -- there's no
    per-tile selector, only the shared Day picker.

    confidence (see update_weather_confidence.py) adds an ensemble p10-p90 band behind the
    selected-day line: how far apart ECMWF's 51 members are for that hour, i.e. how much the
    forecast itself is worth trusting. It's drawn first so it sits behind the lines, and it's
    best-effort -- the ensemble endpoint only serves a few days around now, so most days in
    the picker get an empty (but still present, to keep the trace count fixed) band."""
    default_day_idx = default_day_idx if default_day_idx is not None else default_date_idx
    df_dates = df[time_col].dt.date
    conf_dates = confidence['timestamp'].dt.date if confidence is not None else None
    figs = {}
    for var in var_map:
        fig = go.Figure()
        for di, date in enumerate(DAY_OPTIONS):
            visible = (di == default_day_idx)
            day_z, prev_z, avg_z, prev_date = _reference_series(df, df_dates, var, date)

            lo_x, lo_y, hi_y = [], [], []
            if confidence is not None and f'{var}_p10' in confidence.columns:
                band = confidence[conf_dates == date].sort_values('hour')
                band = band[band[f'{var}_p10'].notna() & band[f'{var}_p90'].notna()]
                if not band.empty:
                    lo_x = band['hour'].tolist()
                    lo_y = band[f'{var}_p10'].tolist()
                    hi_y = band[f'{var}_p90'].tolist()
            fig.add_trace(go.Scatter(
                x=lo_x, y=lo_y, mode='lines', line=dict(width=0),
                hoverinfo='skip', showlegend=False, visible=visible
            ))
            fig.add_trace(go.Scatter(
                x=lo_x, y=hi_y, mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(52,152,219,0.16)', name='Ensemble p10-p90',
                hoverinfo='skip', showlegend=False, visible=visible
            ))

            fig.add_trace(go.Scatter(
                x=avg_z['hour'], y=avg_z[var], name='7d Average', mode='lines',
                line=dict(color=COLORS['avg'], dash='dot', width=1.5), visible=visible
            ))
            fig.add_trace(go.Scatter(
                x=prev_z['hour'], y=prev_z[var], name=str(prev_date), mode='lines+markers',
                line=dict(color=COLORS['prev_day'], dash='dash'), marker=dict(size=5, line=dict(width=1, color=COLORS['ring'])),
                visible=visible
            ))
            fig.add_trace(go.Scatter(
                x=day_z['hour'], y=day_z[var], name=str(date), mode='lines+markers',
                line=dict(color=COLORS['dam'], width=2.5), marker=dict(size=6, line=dict(width=1.5, color=COLORS['ring'])),
                visible=visible
            ))

        fig.update_layout(
            template='plotly_dark',
            title=None,
            showlegend=False,
            xaxis=hour_xaxis(dtick=4, showspikes=True, spikemode='across', spikesnap='cursor',
                              spikedash='dot', spikethickness=1, spikecolor=COLORS['muted'], gridcolor=COLORS['grid']),
            yaxis=dict(gridcolor=COLORS['grid'], hoverformat='.1f'),
            hovermode='x unified',
            margin=dict(t=10, b=30, l=45, r=10),
            height=240
        )
        figs[var] = fig
    return figs


def build_wide_table_fig(df, time_col, var_map, default_var_idx, tab_label, colorscale='Thermal', polished=False):
    """Same rolling-window heatmap as build_table_fig. polished=False keeps the native Plotly
    updatemenu dropdown to pick the variable; polished=True drops it since variable switching
    is then driven externally (the shared day-bar selector, relabeled 'Variable' for this tab)."""
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
            # Plain Python lists, not the numpy/pandas objects themselves: plotly.py encodes
            # numpy-backed numeric arrays as compact base64 ({"dtype":...,"bdata":...}) which
            # copyTableTSV (see generar_web.py) can't read directly off the rendered figure --
            # plain lists always serialize as ordinary JSON arrays.
            z=pivot.values.tolist(), x=list(pivot.columns), y=list(pivot.index),
            text=text, texttemplate='%{text}', textfont=dict(size=11),
            colorscale=colorscale, zmin=zmin, zmax=zmax,
            colorbar=dict(title=unit),
            visible=(i == default_var_idx),
            hovertemplate=f'Date %{{y}}, Hour %{{x}}<br>{label}: %{{z:.1f}} {unit}<extra></extra>'
        ))

    label0 = var_map[var_keys[default_var_idx]][0]
    title = None if polished else f'{tab_label} - Hourly Table - {label0} (last {TABLE_DAYS} days)'
    layout_kwargs = dict(
        template='plotly_dark',
        title=title,
        xaxis_title='Hour', yaxis_title='Date',
        xaxis=dict(dtick=1, side='top'),
        yaxis=dict(tickmode='array', tickvals=SELECTABLE_DATE_STRS, ticktext=SELECTABLE_DATE_STRS),
        margin=dict(t=50, b=40, r=40) if polished else dict(t=90, b=40, r=40),
        height=(100 if polished else 130) + TABLE_DAYS * TABLE_ROW_HEIGHT
    )
    if not polished:
        buttons = [
            dict(label=var_map[var][0], method='update',
                 args=[{'visible': [j == i for j in range(len(var_keys))]},
                       {'title': f'{tab_label} - Hourly Table - {var_map[var][0]} (last {TABLE_DAYS} days)'}])
            for i, var in enumerate(var_keys)
        ]
        layout_kwargs['updatemenus'] = [dict(buttons=buttons, direction='down', x=1.0, y=1.12, xanchor='right', yanchor='top',
                                              active=default_var_idx, showactive=True)]
    fig.update_layout(**layout_kwargs)
    return fig
