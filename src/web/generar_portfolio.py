"""Builds docs/portfolio.html: the trading history from data/historical_pnl.csv (itself
built by parse_reports.py from the IESO participation XML reports in data/reports),
grouped by month -- a Won/Lost/No-exposure breakdown, an hour-by-day case grid, and a
per-trade log. Stats are month-scoped only, on purpose: an all-time accumulated view was
tried and dropped -- the per-month cut is what's actually useful here."""
import json
import os

import pandas as pd
import plotly.graph_objects as go

from theme import COLORS, TABLE_ROW_HEIGHT

os.makedirs('docs', exist_ok=True)

BADGE_CLASS = {'Won': 'badge-won', 'Lost': 'badge-lost', 'Flat': 'badge-flat', 'Pending': 'badge-pending'}

# "What if every missed order had filled?" -- No-exposure rows only record what actually
# cleared (0 MW), not the size that was intended, so there's no way to derive a hypothetical
# fill size from the report itself. 1 MW is the assumption: 91 of this book's 94 real trades
# (97%) are exactly 1 MW, so it's the representative size, not an arbitrary round number.
WHATIF_FILL_MW = 1.0

# Case-grid heatmap: one flat color per outcome (not a magnitude scale) -- the point is the
# category, not the size. 'blank' (an hour never listed in the report at all -- not a case)
# sits right on the panel background so it disappears; 'No exposure' (explicitly 0 MW in the
# report -- submitted, didn't clear) gets its own visible-but-calm gray so it reads as a real,
# counted case instead of empty space. 'Won (hyp.)'/'Lost (hyp.)' are only ever used in the
# what-if overlay (see _grid_cell_arrays) -- same hues as the real outcomes at roughly half
# the opacity, so a hypothetical result is legible as the same category but visibly softer,
# never confusable with a real one.
GRID_CATEGORY = {'blank': 0, 'No exposure': 1, 'Lost': 2, 'Won': 3, 'Pending': 4, 'Flat': 5,
                  'Won (hyp.)': 6, 'Lost (hyp.)': 7}
GRID_COLORS = ['#242424', 'rgba(102,102,102,0.5)', 'rgba(231,76,60,0.55)', 'rgba(46,204,113,0.55)',
               'rgba(232,163,61,0.55)', 'rgba(150,150,150,0.55)', 'rgba(46,204,113,0.28)', 'rgba(231,76,60,0.28)']

GRID_LEGEND = f'''<div class="chart-legend">
  <span><i class="sw" style="background:{GRID_COLORS[3]}"></i>Won</span>
  <span><i class="sw" style="background:{GRID_COLORS[2]}"></i>Lost</span>
  <span><i class="sw" style="background:{GRID_COLORS[1]}"></i>No exposure</span>
  <span><i class="sw" style="background:{GRID_COLORS[4]}"></i>Pending</span>
</div>'''

GRID_LEGEND_WHATIF = f'''<div class="chart-legend">
  <span><i class="sw" style="background:{GRID_COLORS[3]}"></i>Won</span>
  <span><i class="sw" style="background:{GRID_COLORS[2]}"></i>Lost</span>
  <span><i class="sw" style="background:{GRID_COLORS[6]}"></i>Won (hypothetical)</span>
  <span><i class="sw" style="background:{GRID_COLORS[7]}"></i>Lost (hypothetical)</span>
  <span><i class="sw" style="background:{GRID_COLORS[4]}"></i>Pending</span>
</div>'''


def fmt_money(val, signed=False):
    if pd.isna(val):
        return '—'
    sign = '+' if (signed and val > 0) else ''
    return f"{sign}${val:,.2f}"


def outcome_of(energy_mw, pnl):
    """energy_mw == 0 means the report explicitly lists this hour with nothing filled -- an
    order was sent but the price never cleared -- which is its own category, checked before
    pnl (which is trivially 0, or even NaN if the price join failed) says anything else."""
    if energy_mw == 0:
        return 'No exposure'
    if pd.isna(pnl):
        return 'Pending'
    if pnl > 0:
        return 'Won'
    if pnl < 0:
        return 'Lost'
    return 'Flat'


def pnl_cls(val):
    if pd.isna(val) or val == 0:
        return ''
    return 'pos' if val > 0 else 'neg'


def whatif_pnl_of(outcome, position, spread):
    """What a 'No exposure' hour's PnL would have been at WHATIF_FILL_MW, same side and the
    real spread that hour -- None for every other outcome (nothing hypothetical to compute)
    or if spread itself is missing (price join failed for that hour)."""
    if outcome != 'No exposure' or pd.isna(spread):
        return None
    signed_size = WHATIF_FILL_MW if position == 'Virtual Gen' else -WHATIF_FILL_MW
    return signed_size * spread


def load_data():
    """df is every row parse_reports.py wrote -- every hour the report explicitly lists,
    whether it filled (energy_mw != 0) or not (energy_mw == 0, an order that never cleared).
    An hour missing from the report entirely isn't in df at all -- it's not a case, just
    nothing sent. position comes from resource_type (GEN offer = Virtual Gen, LD bid = Virtual
    Load -- the actual IESO terms for the two sides) rather than energy_mw's sign, since a
    0 MW row has no sign to read but still has a real intended side. trades is the subset
    that actually filled, used for the per-trade log."""
    df = pd.read_csv('data/historical_pnl.csv', parse_dates=['date'])
    df['month'] = df['date'].dt.strftime('%Y-%m')
    df['position'] = df['resource_type'].apply(lambda t: 'Virtual Gen' if t == 'GEN' else 'Virtual Load')
    df['size'] = df['energy_mw'].abs()
    df['outcome'] = df.apply(lambda r: outcome_of(r['energy_mw'], r['pnl']), axis=1)
    df['whatif_pnl'] = df.apply(lambda r: whatif_pnl_of(r['outcome'], r['position'], r['spread']), axis=1)
    df = df.sort_values(['date', 'hour'])

    trades = df[df['energy_mw'] != 0].copy()
    return df, trades


def _discrete_category_colorscale(colors):
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale.append([i / n, c])
        scale.append([(i + 1) / n, c])
    return scale


def _grid_cell_arrays(dates, month_df, whatif=False):
    """The z/text/hover arrays build_case_grid_fig plots (real mode) -- also called a second
    time with whatif=True to get the 'what if every missed order had filled' overlay, so the
    two states can be swapped client-side via Plotly.restyle instead of rendering two figures.
    In whatif mode every other cell (real trades, pending, blank) is untouched; only
    'No exposure' cells recolor to the (softer-opacity) hypothetical Won/Lost category, using
    whatif_pnl (same position, real spread for that hour, WHATIF_FILL_MW instead of the 0 MW
    that actually cleared)."""
    date_strs = [f"{d:%Y-%m-%d}" for d in dates]
    n = len(dates)
    row_of = {d: i for i, d in enumerate(dates)}
    TOTAL_COL = 24
    z = [[GRID_CATEGORY['blank'] + 0.5] * 25 for _ in range(n)]
    text = [[''] * 25 for _ in range(n)]
    hover = [[''] * 25 for _ in range(n)]

    for _, r in month_df.iterrows():
        i, j = row_of[r['date']], int(r['hour']) - 1
        if whatif and r['outcome'] == 'No exposure' and pd.notna(r['whatif_pnl']):
            cat = 'Won (hyp.)' if r['whatif_pnl'] > 0 else 'Lost (hyp.)'
            z[i][j] = GRID_CATEGORY[cat] + 0.5
            text[i][j] = f"~{r['whatif_pnl']:+.0f}"
            hover[i][j] = (f"{r['position']} {WHATIF_FILL_MW:g} MW (hypothetical -- this order didn't really clear)"
                            f"<br>What-if PnL: {fmt_money(r['whatif_pnl'], signed=True)}")
            continue
        z[i][j] = GRID_CATEGORY[r['outcome']] + 0.5
        if r['outcome'] == 'No exposure':
            text[i][j] = ''
            hover[i][j] = f"{r['position']} order sent, didn't clear (0 MW)"
            continue
        # Literal "·" (not &middot;): Plotly hover text only decodes a small fixed set of real
        # HTML tags (<br>, <b>, ...), not named entities -- those print as literal "&middot;".
        detail = (f"{r['position']} {r['size']:.1f} MW<br>DAM {fmt_money(r['lmp_dam'])} "
                  f"· RTM {fmt_money(r['lmp_rtm'])} · Spread {fmt_money(r['spread'], signed=True)}")
        if r['outcome'] == 'Pending':
            text[i][j] = '…'
            hover[i][j] = f"{detail}<br>Pending settlement"
        else:
            text[i][j] = '0' if r['outcome'] == 'Flat' else f"{r['pnl']:+.0f}"
            hover[i][j] = f"{detail}<br>PnL: {fmt_money(r['pnl'], signed=True)} ({r['outcome']})"

    # Trailing Total column: the day's net PnL. In real mode that's just settled trades' pnl
    # (No-exposure contributes $0, Pending is excluded -- unknown outcome). In whatif mode a
    # day can have BOTH a real trade and a hypothetical one, and they need to net together --
    # summing a single column (as an earlier version of this did) silently dropped whichever
    # side wasn't in that column (real trades have no whatif_pnl, No-exposure rows have no
    # pnl), understating the total on any day that mixed the two.
    def _effective_pnl(row):
        if row['outcome'] in ('Won', 'Lost', 'Flat'):
            return row['pnl']
        if whatif and row['outcome'] == 'No exposure':
            return row['whatif_pnl']
        return None

    included_outcomes = ['Won', 'Lost', 'Flat'] + (['No exposure'] if whatif else [])
    for d in dates:
        i = row_of[d]
        day_rows = month_df[(month_df['date'] == d) & month_df['outcome'].isin(included_outcomes)]
        if day_rows.empty:
            continue
        day_effective = day_rows.apply(_effective_pnl, axis=1)
        if day_effective.isna().all():
            continue
        day_total = day_effective.sum()
        z[i][TOTAL_COL] = GRID_CATEGORY['Won' if day_total > 0 else ('Lost' if day_total < 0 else 'Flat')] + 0.5
        text[i][TOTAL_COL] = f"{day_total:+.0f}"
        hover[i][TOTAL_COL] = f"Daily total: {fmt_money(day_total, signed=True)}"

    # Trailing "Total" row: the mirror of the Total column above -- summed down each hour
    # across every date in the month instead of across every hour within one date. Its
    # bottom-right cell (Total row x Total column) is the grand total for the month, reached
    # by summing the same rows a third way (by hour instead of by day); it should always match
    # the Net PnL stat tile.
    total_row = [GRID_CATEGORY['blank'] + 0.5] * 25
    total_text = [''] * 25
    total_hover = [''] * 25
    for h in range(1, 25):
        hour_rows = month_df[(month_df['hour'] == h) & month_df['outcome'].isin(included_outcomes)]
        if hour_rows.empty:
            continue
        hour_effective = hour_rows.apply(_effective_pnl, axis=1)
        if hour_effective.isna().all():
            continue
        hour_total = hour_effective.sum()
        col = h - 1
        total_row[col] = GRID_CATEGORY['Won' if hour_total > 0 else ('Lost' if hour_total < 0 else 'Flat')] + 0.5
        total_text[col] = f"{hour_total:+.0f}"
        total_hover[col] = f"HE{h:02d} total: {fmt_money(hour_total, signed=True)}"

    month_rows = month_df[month_df['outcome'].isin(included_outcomes)]
    month_effective = month_rows.apply(_effective_pnl, axis=1)
    if month_effective.notna().any():
        month_total = month_effective.sum()
        total_row[TOTAL_COL] = GRID_CATEGORY['Won' if month_total > 0 else ('Lost' if month_total < 0 else 'Flat')] + 0.5
        total_text[TOTAL_COL] = f"{month_total:+.0f}"
        total_hover[TOTAL_COL] = f"Month total: {fmt_money(month_total, signed=True)}"

    z.append(total_row)
    text.append(total_text)
    hover.append(total_hover)

    return z, text, hover


def build_case_grid_fig(dates, z, text, hover):
    """One row per report date that month plus a trailing Total row, one column per hour
    (HE01-HE24) plus a trailing Total column, colored by outcome instead of magnitude -- every
    case at a glance. z/text/hover come from _grid_cell_arrays (real mode) -- the caller builds
    them separately instead of this function doing it internally so the same call can also
    produce the what-if arrays without wiring a second Heatmap trace just to get at them; that
    also means z/text/hover already carry the trailing Total row/column baked in, so this
    function just needs to grow date_strs by one label to line up with them.
    yaxis.type is forced to 'category': dates are strings that look like dates, and Plotly's
    default type inference reads them as a real date axis, spacing rows by elapsed calendar
    time instead of evenly -- rows for report dates separated by a weekend or a gap end up
    stretched apart. The other hourly tables never hit this because they show a contiguous
    day range with no gaps; report dates are inherently sparse."""
    row_labels = [f"{d:%Y-%m-%d}" for d in dates] + ['Total']
    n = len(dates)

    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(1, 26)), y=row_labels,
        text=text, texttemplate='%{text}', textfont=dict(size=10, color='#eee'),
        customdata=hover, hovertemplate='%{y}, Hour %{x}<br>%{customdata}<extra></extra>',
        colorscale=_discrete_category_colorscale(GRID_COLORS), zmin=0, zmax=len(GRID_COLORS),
        showscale=False, xgap=2, ygap=2,
    ))
    fig.add_vline(x=24.5, line_color=COLORS['muted'], line_width=1)
    fig.add_hline(y=n - 0.5, line_color=COLORS['muted'], line_width=1)
    fig.update_layout(
        template='plotly_dark', title=None,
        xaxis=dict(tickmode='array', tickvals=list(range(1, 25)) + [25], ticktext=[str(h) for h in range(1, 25)] + ['Total'],
                   range=[0.5, 25.5], side='top', gridcolor=COLORS['grid']),
        yaxis=dict(type='category', tickmode='array', tickvals=row_labels, ticktext=row_labels,
                   autorange='reversed'),
        margin=dict(t=30, b=10, l=90, r=10),
        height=40 + (n + 1) * TABLE_ROW_HEIGHT,
    )
    return fig


def stat_tile(label, value, cls='', sub=None, extra_cls=''):
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ''
    return f'''<div class="stat-tile {extra_cls}"><div class="stat-label">{label}</div>
    <div class="stat-value {cls}">{value}</div>{sub_html}</div>'''


def render_trade_row(r):
    pos_cls = 'badge-long' if r['position'] == 'Virtual Gen' else 'badge-short'
    out_cls = BADGE_CLASS[r['outcome']]
    return f'''<tr>
  <td>{r['date']:%Y-%m-%d}</td>
  <td>HE{int(r['hour']):02d}</td>
  <td><span class="badge {pos_cls}">{r['position']}</span></td>
  <td>{r['size']:.1f} MW</td>
  <td>{fmt_money(r['lmp_dam'])}</td>
  <td>{fmt_money(r['lmp_rtm'])}</td>
  <td>{fmt_money(r['spread'], signed=True)}</td>
  <td class="{pnl_cls(r['pnl'])}">{fmt_money(r['pnl'], signed=True)}</td>
  <td><span class="badge {out_cls}">{r['outcome']}</span></td>
</tr>'''


def build_portfolio():
    df, trades = load_data()
    if trades.empty:
        print("No trades found in historical_pnl.csv (run parse_reports.py first).")
        return

    months = sorted(df['month'].unique(), reverse=True)
    tabs_html, sections_html = '', ''
    whatif_grid_data = {}
    for i, m in enumerate(months):
        active = 'active' if i == 0 else ''
        month_label = pd.to_datetime(m + '-01').strftime('%B %Y')
        tabs_html += f'<button class="tab-btn {active}" onclick="showMonth(\'{m}\', this)">{month_label}</button>\n'

        m_df = df[df['month'] == m]
        m_dates = sorted(m_df['date'].unique(), reverse=True)
        m_trades = trades[trades['month'] == m].sort_values(['date', 'hour'], ascending=[False, True])
        counts = m_df['outcome'].value_counts()
        m_won, m_lost = int(counts.get('Won', 0)), int(counts.get('Lost', 0))
        m_pending = int(counts.get('Pending', 0))
        m_no_exposure = int(counts.get('No exposure', 0))
        m_net = m_trades.loc[m_trades['outcome'] != 'Pending', 'pnl'].sum() if len(m_trades) else 0
        m_total_cases = len(m_df)

        def _pct(count):
            return f"{count} ({count / m_total_cases:.0%})" if m_total_cases else str(count)

        month_tiles = stat_tile('Won', _pct(m_won), cls='pos')
        month_tiles += stat_tile('Lost', _pct(m_lost), cls='neg')
        month_tiles += stat_tile('No exposure', _pct(m_no_exposure))
        month_tiles += stat_tile('Net PnL', fmt_money(m_net, signed=True), cls=pnl_cls(m_net), extra_cls='highlight')
        if m_pending:
            month_tiles += stat_tile('Pending', _pct(m_pending))

        # What-if: every 'No exposure' hour recategorized by its hypothetical PnL's sign
        # (whatif_pnl_of, computed in load_data). Rows where whatif_pnl is None (spread was
        # missing for that hour) stay uncategorized -- same 'No exposure' bucket, just
        # relabeled so it's clear a couple of hours simply couldn't be projected.
        no_exp_rows = m_df[m_df['outcome'] == 'No exposure']
        wf_won = m_won + int((no_exp_rows['whatif_pnl'] > 0).sum())
        wf_lost = m_lost + int((no_exp_rows['whatif_pnl'] < 0).sum())
        wf_undetermined = int(no_exp_rows['whatif_pnl'].isna().sum())
        wf_net = m_net + no_exp_rows['whatif_pnl'].sum()

        whatif_tiles = stat_tile('Won', _pct(wf_won), cls='pos')
        whatif_tiles += stat_tile('Lost', _pct(wf_lost), cls='neg')
        whatif_tiles += stat_tile('No exposure', _pct(wf_undetermined),
                                   sub='not enough data to project' if wf_undetermined else None)
        whatif_tiles += stat_tile('Net PnL', fmt_money(wf_net, signed=True), cls=pnl_cls(wf_net), extra_cls='highlight',
                                   sub=f'hypothetical -- assumes every missed order filled at {WHATIF_FILL_MW:g} MW')
        if m_pending:
            whatif_tiles += stat_tile('Pending', _pct(m_pending))

        z_real, text_real, hover_real = _grid_cell_arrays(m_dates, m_df, whatif=False)
        z_wf, text_wf, hover_wf = _grid_cell_arrays(m_dates, m_df, whatif=True)
        grid_fig = build_case_grid_fig(m_dates, z_real, text_real, hover_real)
        whatif_grid_data[m] = {
            'real': {'z': z_real, 'text': text_real, 'customdata': hover_real},
            'whatif': {'z': z_wf, 'text': text_wf, 'customdata': hover_wf},
        }
        rows_html = '\n'.join(render_trade_row(r) for _, r in m_trades.iterrows())

        sections_html += f'''<div id="month-{m}" class="tab-content {active}">
  <label class="whatif-toggle">
    <input type="checkbox" onchange="toggleWhatIf('{m}', this.checked)">
    What if every missed order had filled? <span class="stat-sub">(assumes {WHATIF_FILL_MW:g} MW each, same side, real market spread -- see hover for detail)</span>
  </label>
  <div class="stat-row month-stats" id="stats-real-{m}">{month_tiles}</div>
  <div class="stat-row month-stats hidden" id="stats-whatif-{m}">{whatif_tiles}</div>
  <div id="legend-real-{m}">{GRID_LEGEND}</div>
  <div id="legend-whatif-{m}" class="hidden">{GRID_LEGEND_WHATIF}</div>
  <div class="table-container grid-container">
    {grid_fig.to_html(full_html=False, include_plotlyjs=('cdn' if i == 0 else False), div_id=f'grid-{m}')}
  </div>
  <div class="table-container" style="margin-top:16px;">
    <table class="trade-table">
      <thead><tr><th>Date</th><th>Hour</th><th>Position</th><th>Size</th><th>DAM</th><th>RTM</th><th>Spread</th><th>PnL</th><th>Outcome</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>\n'''

    html = f"""<html>
<head>
<meta charset="utf-8">
<title>Portfolio</title>
<style>
  body {{ background:{COLORS['ring']}; color:#eee; margin:0; padding:24px;
    font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
  a {{ color:{COLORS['dam']}; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .subtitle {{ color:#888; font-size:13px; margin:0 0 20px; max-width:760px; }}
  .back-link {{ display:inline-block; margin-bottom:16px; color:#aaa; text-decoration:none; font-size:13px; }}
  .back-link:hover {{ color:#fff; }}

  .stat-row {{ display:flex; flex-wrap:wrap; gap:16px; margin:0 0 16px; }}
  .stat-tile {{ background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:12px 20px; flex:1; min-width:130px; }}
  .stat-tile.highlight {{ border-color:#555; background:#202020; }}
  .stat-label {{ color:#888; font-size:12px; }}
  .stat-value {{ color:#eee; font-size:22px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }}
  .stat-value.pos {{ color:{COLORS['positive']}; }}
  .stat-value.neg {{ color:{COLORS['negative']}; }}
  .stat-sub {{ color:#777; font-size:11px; margin-top:2px; }}
  .month-stats .stat-tile {{ padding:10px 16px; }}
  .month-stats .stat-value {{ font-size:18px; }}

  .tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin:0 0 16px; border-bottom:1px solid #333; padding-bottom:8px; }}
  .tab-btn {{ background:#1e1e1e; color:#ccc; border:1px solid #333; border-top:3px solid {COLORS['dam']};
    border-radius:6px 6px 0 0; padding:8px 16px; cursor:pointer; font-size:13px; }}
  .tab-btn.active {{ background:#2c2c2c; color:#fff; border-bottom:2px solid {COLORS['dam']}; }}

  /* max-height:0 (instead of display:none) keeps the container's width intact so Plotly's
     auto-sizing doesn't collapse hidden tabs' charts to a fallback width on first render --
     same fix already used for the main dashboard's tabs (see generar_web.py). */
  .tab-content {{ max-height:0; overflow:hidden; }}
  .tab-content.active {{ max-height:none; }}

  .chart-legend {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; font-size:12px; color:#aaa; margin:0 0 8px; }}
  .chart-legend .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }}
  .grid-container {{ padding:4px 8px; }}

  .hidden {{ display:none; }}
  .whatif-toggle {{ display:flex; align-items:center; gap:8px; margin:0 0 12px; font-size:13px; color:#ccc; cursor:pointer; }}
  .whatif-toggle input {{ cursor:pointer; }}
  .whatif-toggle .stat-sub {{ margin:0; }}

  .table-container {{ overflow-x:auto; max-width:100%; border:1px solid #333; border-radius:8px; }}
  .trade-table {{ border-collapse:collapse; width:100%; background:#1a1a1a; font-size:13px; }}
  .trade-table th, .trade-table td {{ padding:7px 12px; text-align:right; border-bottom:1px solid #2a2a2a; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .trade-table th {{ background:#222; color:#888; font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:0.4px; position:sticky; top:0; z-index:1; }}
  .trade-table td:nth-child(1), .trade-table td:nth-child(2), .trade-table td:nth-child(3),
  .trade-table th:nth-child(1), .trade-table th:nth-child(2), .trade-table th:nth-child(3) {{ text-align:left; }}
  .trade-table tbody tr:hover td {{ background:#222; }}
  .trade-table td.pos {{ color:{COLORS['positive']}; font-weight:600; }}
  .trade-table td.neg {{ color:{COLORS['negative']}; font-weight:600; }}

  .badge {{ display:inline-block; padding:2px 9px; border-radius:10px; font-size:11px; font-weight:600; white-space:nowrap; }}
  .badge-long {{ background:rgba(52,152,219,0.15); color:{COLORS['dam']}; }}
  .badge-short {{ background:rgba(230,126,34,0.15); color:{COLORS['rtm']}; }}
  .badge-won {{ background:rgba(46,204,113,0.15); color:{COLORS['positive']}; }}
  .badge-lost {{ background:rgba(231,76,60,0.15); color:{COLORS['negative']}; }}
  .badge-flat {{ background:rgba(232,163,61,0.15); color:{COLORS['prev_day']}; }}
  .badge-pending {{ background:rgba(255,255,255,0.08); color:#888; }}
</style>
</head>
<body>

<a class="back-link" href="index.html">&larr; Back to dashboard</a>
<h1>Portfolio</h1>
<p class="subtitle">Monthly results for every LRG bid/offer submitted to the IESO market, built from the
participation reports in data/reports. Virtual Gen profits when DAM clears above RTM; Virtual Load profits
when RTM clears above DAM. The case grid shows every hour we could have traded that month, not just the
ones we did.</p>

<div class="tabs">
  {tabs_html}
</div>

{sections_html}

<script>
function showMonth(month, btn) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.getElementById('month-' + month).classList.add('active');
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    btn.classList.add('active');
}}

// {{month: {{real: {{z, text, customdata}}, whatif: {{...}}}}}} -- both states for every
// month's case grid, swapped via Plotly.restyle instead of rendering two figures.
const WHATIF_GRID = {json.dumps(whatif_grid_data)};

function toggleWhatIf(month, on) {{
    document.getElementById('stats-real-' + month).classList.toggle('hidden', on);
    document.getElementById('stats-whatif-' + month).classList.toggle('hidden', !on);
    document.getElementById('legend-real-' + month).classList.toggle('hidden', on);
    document.getElementById('legend-whatif-' + month).classList.toggle('hidden', !on);
    const data = WHATIF_GRID[month][on ? 'whatif' : 'real'];
    Plotly.restyle('grid-' + month, {{z: [data.z], text: [data.text], customdata: [data.customdata]}}, [0]);
}}
</script>
</body>
</html>
"""

    with open('docs/portfolio.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("Portfolio generated successfully at docs/portfolio.html!")


if __name__ == '__main__':
    build_portfolio()
