"""Shared helpers and JSON constants used across dashboard tab builders."""

import json

import pandas as pd

from dashboard_data import (
    DAY_OPTION_STRS,
    DAY_OPTIONS,
    LOAD_VARS,
    load_var_keys,
    weather_label,
    weather_var_keys,
    wind_zones,
    zones,
)

ZONES_JSON = json.dumps(zones)
DATES_JSON = json.dumps(DAY_OPTION_STRS)
WEATHER_LABELS_JSON = json.dumps([weather_label(v) for v in weather_var_keys])

LOAD_LABELS_JSON = json.dumps([LOAD_VARS[v][0] for v in load_var_keys])
LOAD_Y_AXIS_TITLES = [f'{LOAD_VARS[v][0]} ({LOAD_VARS[v][1]})' for v in load_var_keys]

WIND_ZONES_JSON = json.dumps(wind_zones)


def date_input_html(elem_id, onchange, selected_date_str):
    """Native browser date picker instead of a long <option> list: same YYYY-MM-DD value
    format the JS already expects, so applyFigSelection/applyAllFigs need no changes.
    min/max clamp it to the actual selectable window (DAY_OPTION_STRS is one contiguous
    range, so bounding it is enough -- no gaps to worry about)."""
    return (f'<input type="date" id="{elem_id}" onchange="{onchange}" '
            f'min="{DAY_OPTION_STRS[0]}" max="{DAY_OPTION_STRS[-1]}" value="{selected_date_str}">')


def zone_options_html(options, default):
    return '\n'.join(
        f'<option value="{o}"{" selected" if o == default else ""}>{o}</option>'
        for o in options
    )


def register_fig(div_id, traces_per_combo, title_prefix, zones_json=None, y_axis_titles=None, date_sel_id=None,
                 show_title=True, zone_sel_id=None, compact_zone_data=None, default_zone_idx=None, compact_kind=None):
    zones_json = zones_json if zones_json is not None else ZONES_JSON
    y_axis_json = json.dumps(y_axis_titles) if y_axis_titles else 'null'
    date_sel_json = json.dumps(date_sel_id) if date_sel_id else 'null'
    zone_sel_json = json.dumps(zone_sel_id) if zone_sel_id else 'null'
    compact_json = json.dumps(compact_zone_data) if compact_zone_data else 'null'
    default_zone_json = json.dumps(default_zone_idx) if default_zone_idx is not None else 'null'
    compact_kind_json = json.dumps(compact_kind) if compact_kind else 'null'
    return (f"<script>registerFig('{div_id}', {zones_json}, {DATES_JSON}, {traces_per_combo}, "
            f"'{title_prefix}', {y_axis_json}, {date_sel_json}, {json.dumps(show_title)}, {zone_sel_json}, "
            f"{compact_json}, {default_zone_json}, {compact_kind_json});</script>")


def zone_hourly_data(df, zones_list, exclude_zone, location_col='location', value_col='lmp'):
    """Compact {zone: {date_str: [24 hourly values]}} for every zone except exclude_zone (the
    one build_hourly_fig(compact_zones=True) already pre-bakes as full Plotly traces). Read by
    the page's JS (see showCompactZone) to recompute a chart on the fly when the zone selector
    picks a non-default zone, instead of pre-baking every zone x day combination. Covers the
    Day picker's window plus 7 extra trailing days so the earliest selectable day can still
    compute its 7d-average reference."""
    window_start = DAY_OPTIONS[0] - pd.Timedelta(days=7)
    df_t = df[(df['interval_start_local'].dt.date >= window_start) &
              (df['interval_start_local'].dt.date <= DAY_OPTIONS[-1])].copy()
    df_t['date'] = df_t['interval_start_local'].dt.date.astype(str)
    data = {}
    for zone in zones_list:
        if zone == exclude_zone:
            continue
        zdf = df_t[df_t[location_col] == zone]
        data[zone] = {
            date: [None if pd.isna(v) else round(float(v), 2) for v in g.set_index('hour')[value_col].reindex(range(1, 25))]
            for date, g in zdf.groupby('date')
        }
    return data


def hour_ranges(hours):
    """[12,13,14,20,21] -> 'HE12-14, HE20-21'."""
    hours = sorted(hours)
    out, start, prev = [], None, None
    for h in hours + [None]:
        if start is None:
            start = prev = h
        elif h is not None and h == prev + 1:
            prev = h
        else:
            out.append(f'HE{start}' if start == prev else f'HE{start}-{prev}')
            start = prev = h
    return ', '.join(out)


def zid(base, zone):
    """Element id of one zone's copy of a forecast-tab element: each predict tab holds one panel
    per zone (see zone_panels), and the page JS finds the shown one as base + '--' + zone."""
    return f'{base}--{zone}'


def zone_panels(panels, tab_id, button_label):
    """Wraps {zone: panel_html} into a predict tab: one .zone-panel per zone, the first shown;
    the day-bar's Zone select swaps them (showZonePanel in the page JS). Returns
    (tab_button_html, tab_content_html), both '' when no zone has the files yet."""
    if not any(panels.values()):
        return '', ''
    # A zone just added to ZONES has no live files until its first daily run: say so rather
    # than leave the tab blank when the Zone select lands on it.
    panels = {z: h or f'<h2>{button_label} - {z}</h2><p class="caveat">No {button_label} for {z} yet: it starts '
                       f'with the next 9:00 daily run.</p>' for z, h in panels.items()}
    first = next(iter(panels))
    body = ''.join(f'<div class="zone-panel{" active" if z == first else ""}" data-zone="{z}">{h}</div>'
                   for z, h in panels.items())
    return (f'<button class="tab-btn group-predict" onclick="showTab(\'{tab_id}\', this)">{button_label}</button>',
            f'<div id="tab-{tab_id}" class="tab-content">{body}</div>')


def missing_cols_label(meta):
    """' (rtm_today_early_mean x24, ...)': which inputs were missing, from the meta's
    missing_input_cols (older metas don't have it). The same-day RT summary missing means the
    9:00 RT download failed, which is worth reading differently from a feed not out yet."""
    cols = meta.get('missing_input_cols') or {}
    if not cols:
        return ''
    shown = sorted(cols.items(), key=lambda kv: -kv[1])[:4]
    more = f', +{len(cols) - len(shown)} more' if len(cols) > len(shown) else ''
    return ' (' + ', '.join(f'{c} ×{n}' for c, n in shown) + more + ')'


def vintage_label(meta):
    """Which run produced the live forecast (see forecast_common.VINTAGES)."""
    v = meta.get('vintage')
    rebuilt = ' · walk-forward reconstruction, not a live run' if meta.get('reconstruction') or meta.get('backfilled') else ''
    if v == 'post_dam':
        return ' · post-DAM run (real DAM as input)' + rebuilt
    if v == 'pre_dam':
        return ' · pre-DAM run (DAM input is itself a forecast)' + rebuilt
    return rebuilt
