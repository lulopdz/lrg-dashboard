"""Market tabs (DAM, RTM, Spread) HTML and combo data builders."""

import json

from dashboard_data import DEFAULT_ZONE, TABLE_DAYS, dam, rtm, spread, zones
from dashboard_figures import (
    build_hourly_fig,
    build_spread_detail_fig,
    build_table_fig,
    compact_combos,
)
from tabs.common import ZONES_JSON, register_fig, zone_hourly_data


def build_market_tabs():
    """Build the DAM, RTM, and Spread tab HTML, combo traces, and compact zone JSON."""
    dam_hourly_fig = build_hourly_fig(dam, 'DAM', polished=True, compact_zones=True)
    dam_table_fig = build_table_fig(dam, 'DAM', palette='Blues', polished=True)

    rtm_hourly_fig = build_hourly_fig(rtm, 'RTM', polished=True, compact_zones=True)
    rtm_table_fig = build_table_fig(rtm, 'RTM', palette='Oranges', polished=True)

    compact_zone_data_json = json.dumps({
        'dam': zone_hourly_data(dam, zones, DEFAULT_ZONE),
        'rtm': zone_hourly_data(rtm, zones, DEFAULT_ZONE),
    })

    spread_hourly_fig = build_spread_detail_fig(polished=True, compact_zones=True)
    spread_table_fig = build_table_fig(spread, 'Spread (DAM - RTM)', diverging=True, polished=True)

    combos = {
        'dam-hourly': compact_combos(dam_hourly_fig, 3, n_extra=3),
        'rtm-hourly': compact_combos(rtm_hourly_fig, 3, n_extra=3),
        'spread-hourly': compact_combos(spread_hourly_fig, 3, n_extra=3),
    }

    html = f"""<div id="tab-dam" class="tab-content active">
<div class="card">
  <div class="section-header">
    <h2>Hourly Profile</h2>
  </div>
  {dam_hourly_fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='dam-hourly')}
  {register_fig('dam-hourly', 3, 'DAM - Hourly Profile', show_title=False, zone_sel_id='tab-zone-select', compact_zone_data='dam', default_zone_idx=zones.index(DEFAULT_ZONE))}
</div>
<div class="card">
  <div class="section-header">
    <h2>Hourly Table (last {TABLE_DAYS} days)</h2>
    <button class="copy-btn" onclick="copyTableTSV(this, 'dam-table')">Copy</button>
  </div>
  {dam_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='dam-table')}
  <script>registerTable('dam-table', {ZONES_JSON});</script>
</div>
</div>

<div id="tab-rtm" class="tab-content">
<div class="card">
  <div class="section-header">
    <h2>Hourly Profile</h2>
  </div>
  {rtm_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='rtm-hourly')}
  {register_fig('rtm-hourly', 3, 'RTM - Hourly Profile', show_title=False, zone_sel_id='tab-zone-select', compact_zone_data='rtm', default_zone_idx=zones.index(DEFAULT_ZONE))}
</div>
<div class="card">
  <div class="section-header">
    <h2>Hourly Table (last {TABLE_DAYS} days)</h2>
    <button class="copy-btn" onclick="copyTableTSV(this, 'rtm-table')">Copy</button>
  </div>
  {rtm_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='rtm-table')}
  <script>registerTable('rtm-table', {ZONES_JSON});</script>
</div>
</div>

<div id="tab-spread" class="tab-content">
<div class="card">
  <div class="section-header">
    <h2>Hourly Profile (Positive = green, Negative = red)</h2>
  </div>
  {spread_hourly_fig.to_html(full_html=False, include_plotlyjs=False, div_id='spread-hourly')}
  {register_fig('spread-hourly', 3, 'Spread (DAM - RTM) - Hourly Profile', show_title=False, zone_sel_id='tab-zone-select', compact_zone_data='spread', default_zone_idx=zones.index(DEFAULT_ZONE), compact_kind='spread')}
</div>
<div class="card">
  <div class="section-header">
    <h2>Hourly Table (last {TABLE_DAYS} days)</h2>
    <button class="copy-btn" onclick="copyTableTSV(this, 'spread-table')">Copy</button>
  </div>
  {spread_table_fig.to_html(full_html=False, include_plotlyjs=False, div_id='spread-table')}
  <script>registerTable('spread-table', {ZONES_JSON});</script>
</div>
</div>"""

    return html, combos, compact_zone_data_json
