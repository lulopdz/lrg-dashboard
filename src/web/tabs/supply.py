"""Supply Mix and Adequacy tab HTML, stat tiles, and combo data builders."""

import json

from dashboard_data import DAY_OPTIONS, SUPPLY_MIX, adequacy, default_forecast_date_idx
from dashboard_figures import build_adequacy_grid_figs, build_supply_mix_fig, compact_combos
from tabs.common import register_fig

SUPPLY_TILE_LABELS = ['Peak available supply', 'Tightest capacity margin',
                      'Capacity on outage', 'Peak renewable output']


def supply_mix_tile_data():
    """{date_str: [{value, sub}, ...]} for every day in the picker, so the tiles can track the
    Day selector instead of being frozen on the day the page happened to be generated. They
    used to be baked for one day and silently kept showing it while the charts moved, which
    read as the numbers simply being wrong.

    Every figure here comes from columns the report publishes ahead, so future days are
    covered. A renewable *share* tile used to sit here; its denominator needs scheduled
    output, which future days don't have -- hence renewable output in MW instead of an
    invented percentage."""
    fuels = [c.rsplit('_', 1)[0] for c in SUPPLY_MIX]
    outage_cols = [c for c in ('nuclear_outages', 'gas_outages', 'hydro_outages',
                              'wind_outages', 'solar_outages') if c in adequacy.columns]
    renew_cols = [c for c in ('hydro_forecasted_mwh', 'wind_forecasted', 'solar_forecasted')
                  if c in adequacy.columns]
    adq_dates = adequacy['interval_start_local'].dt.date

    def tile(value, sub):
        return {'value': value, 'sub': sub}

    out = {}
    for date in DAY_OPTIONS:
        day = adequacy[adq_dates == date]
        if day.empty:
            continue
        entries = []

        parts = [(day[f'{f}_capacity'] - day[f'{f}_outages']) for f in fuels
                 if f'{f}_capacity' in day.columns and f'{f}_outages' in day.columns]
        total = sum(p.fillna(0) for p in parts) if parts else None
        entries.append(tile(f'{total.max():,.0f} MW', f'at HE{int(day.loc[total.idxmax(), "hour"]):02d}')
                       if total is not None and total.notna().any() else None)

        margin = day['capacity_excess_shortfall'] if 'capacity_excess_shortfall' in day.columns else None
        entries.append(tile(f'{margin.min():,.0f} MW', f'at HE{int(day.loc[margin.idxmin(), "hour"]):02d}')
                       if margin is not None and margin.notna().any() else None)

        outage = day[outage_cols].sum(axis=1) if outage_cols else None
        entries.append(tile(f'{outage.mean():,.0f} MW', 'daily average')
                       if outage is not None and outage.notna().any() else None)

        renew = day[renew_cols].sum(axis=1) if renew_cols else None
        entries.append(tile(f'{renew.max():,.0f} MW', f'at HE{int(day.loc[renew.idxmax(), "hour"]):02d}')
                       if renew is not None and renew.notna().any() else None)

        if any(entries):
            out[str(date)] = entries
    return out


def supply_mix_stat_tiles():
    """Empty shells; updateSupplyTiles() fills them from SUPPLY_TILE_DATA for the selected day."""
    return '\n'.join(
        f'<div class="stat-tile" id="supply-tile-{i}"><div class="stat-label">{label}</div>'
        f'<div class="stat-value"></div><div class="stat-sub"></div></div>'
        for i, label in enumerate(SUPPLY_TILE_LABELS)
    )


def adequacy_grid_html(adequacy_grid):
    """The three forward-looking adequacy charts, three across, same small-multiples layout as
    the Weather tab's grid."""
    if not adequacy_grid:
        return ''
    tiles = []
    for key, (title, fig, traces) in adequacy_grid.items():
        div_id = f'adequacy-{key}'
        tiles.append(f"""<div class="weather-tile">
  <h3>{title}</h3>
  {fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)}
  {register_fig(div_id, traces, title, zones_json=json.dumps([title]), show_title=False)}
</div>""")
    return f'<div class="weather-grid">\n{chr(10).join(tiles)}\n</div>'


def build_supply_tab():
    """Build the Supply Mix tab button, HTML, combos, and tile data JSON."""
    # Optional: absent on a checkout whose data/ predates update_adequacy.py. Everything that
    # renders the tab is guarded on this being non-None, same as the forecast tabs are guarded on
    # their predict_*.py having run.
    #
    # The headline chart plots available capacity (capacity - outages), which the report publishes
    # ~34 days ahead, so this tab can open on tomorrow like the rest of the forecast group. The
    # *_scheduled columns would have been the more natural choice but only exist once the day-ahead
    # market has run -- they're in the grid below instead, where being blank for future days is
    # expected rather than looking broken. The report's *_offered columns are null for every fuel
    # on future days, so they aren't an option either.
    supply_mix_fig = (build_supply_mix_fig(adequacy, SUPPLY_MIX, mode='available',
                                           default_day_idx=default_forecast_date_idx)
                      if adequacy is not None else None)
    supply_mix_tab_button = ('<button class="tab-btn group-forecast" onclick="showTab(\'supply\', this)">'
                             'Supply Mix</button>' if supply_mix_fig is not None else '')
    # Same default day as the headline chart and as TAB_DATES['supply']: showTab only writes the
    # Day picker's value, it doesn't restyle the figures, so a chart pre-baked on a different day
    # would sit there contradicting the picker until something moved it.
    adequacy_grid = (build_adequacy_grid_figs(adequacy, SUPPLY_MIX, default_day_idx=default_forecast_date_idx)
                     if adequacy is not None else None)

    combos = {}
    if supply_mix_fig is not None:
        combos['supply-mix'] = compact_combos(supply_mix_fig, len(SUPPLY_MIX) + 1)
    for _key, (_title, _fig, _traces) in (adequacy_grid or {}).items():
        combos[f'adequacy-{_key}'] = compact_combos(_fig, _traces)

    supply_tile_data_json = json.dumps(supply_mix_tile_data() if adequacy is not None else {})
    supply_tile_labels_json = json.dumps(SUPPLY_TILE_LABELS)

    supply_mix_tab_html = f"""
<div id="tab-supply" class="tab-content">
<div class="stat-row">
{supply_mix_stat_tiles()}
</div>
<div class="card">
  <div class="section-header">
    <h2>Available supply by fuel</h2>
  </div>
  {supply_mix_fig.to_html(full_html=False, include_plotlyjs=False, div_id='supply-mix')}
  {register_fig('supply-mix', len(SUPPLY_MIX) + 1, 'Supply Mix', zones_json=json.dumps(['Supply Mix']), show_title=False)}
</div>
{adequacy_grid_html(adequacy_grid)}
</div>
""" if supply_mix_fig is not None else ''

    return (
        supply_mix_tab_button,
        supply_mix_tab_html,
        combos,
        supply_tile_data_json,
        supply_tile_labels_json,
        supply_mix_fig is not None,
    )
