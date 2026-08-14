"""Builds docs/portfolio.html: the trading history from data/historical_pnl.csv (itself
built by parse_reports.py from the IESO participation XML reports in data/reports),
grouped by month -- a Won/Lost/No-exposure breakdown, an hour-by-day case grid, and a
per-trade log. Stats are month-scoped only, on purpose: an all-time accumulated view was
tried and dropped -- the per-month cut is what's actually useful here."""
import os

import pandas as pd
import plotly.graph_objects as go

from dashboard_data import COLORS
from dashboard_figures import TABLE_ROW_HEIGHT

os.makedirs('docs', exist_ok=True)

BADGE_CLASS = {'Won': 'badge-won', 'Lost': 'badge-lost', 'Flat': 'badge-flat', 'Pending': 'badge-pending'}

# Case-grid heatmap: one flat color per outcome (not a magnitude scale) -- the point is the
# category, not the size. 'blank' (an hour never listed in the report at all -- not a case)
# sits right on the panel background so it disappears; 'No exposure' (explicitly 0 MW in the
# report -- submitted, didn't clear) gets its own visible-but-calm gray so it reads as a real,
# counted case instead of empty space.
GRID_CATEGORY = {'blank': 0, 'No exposure': 1, 'Lost': 2, 'Won': 3, 'Pending': 4, 'Flat': 5}
GRID_COLORS = ['#242424', 'rgba(102,102,102,0.5)', 'rgba(231,76,60,0.55)', 'rgba(46,204,113,0.55)',
               'rgba(232,163,61,0.55)', 'rgba(150,150,150,0.55)']

GRID_LEGEND = f'''<div class="chart-legend">
  <span><i class="sw" style="background:{GRID_COLORS[3]}"></i>Won</span>
  <span><i class="sw" style="background:{GRID_COLORS[2]}"></i>Lost</span>
  <span><i class="sw" style="background:{GRID_COLORS[1]}"></i>No exposure</span>
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


def build_case_grid_fig(dates, month_df):
    """One row per report date that month, one column per hour (HE01-HE24) plus a trailing
    Total column, colored by outcome instead of magnitude -- every case at a glance. month_df
    is every row the report lists for that month (blank/uninvolved hours just aren't in it,
    and stay the near-invisible 'blank' color); this is what makes 'no exposure' a first-class,
    visible category instead of just a subtracted count in a stat tile.
    yaxis.type is forced to 'category': dates are strings that look like dates, and Plotly's
    default type inference reads them as a real date axis, spacing rows by elapsed calendar
    time instead of evenly -- rows for report dates separated by a weekend or a gap end up
    stretched apart. The other hourly tables never hit this because they show a contiguous
    day range with no gaps; report dates are inherently sparse."""
    date_strs = [f"{d:%Y-%m-%d}" for d in dates]
    n = len(dates)
    row_of = {d: i for i, d in enumerate(dates)}
    TOTAL_COL = 24
    z = [[GRID_CATEGORY['blank'] + 0.5] * 25 for _ in range(n)]
    text = [[''] * 25 for _ in range(n)]
    hover = [[''] * 25 for _ in range(n)]

    for _, r in month_df.iterrows():
        i, j = row_of[r['date']], int(r['hour']) - 1
        z[i][j] = GRID_CATEGORY[r['outcome']] + 0.5
        if r['outcome'] == 'No exposure':
            text[i][j] = ''
            hover[i][j] = f"{r['position']} order sent, didn't clear (0 MW)"
            continue
        detail = (f"{r['position']} {r['size']:.1f} MW<br>DAM {fmt_money(r['lmp_dam'])} "
                  f"&middot; RTM {fmt_money(r['lmp_rtm'])} &middot; Spread {fmt_money(r['spread'], signed=True)}")
        if r['outcome'] == 'Pending':
            text[i][j] = '…'
            hover[i][j] = f"{detail}<br>Pending settlement"
        else:
            text[i][j] = '0' if r['outcome'] == 'Flat' else f"{r['pnl']:+.0f}"
            hover[i][j] = f"{detail}<br>PnL: {fmt_money(r['pnl'], signed=True)} ({r['outcome']})"

    # Trailing Total column: the day's net settled PnL (No-exposure rows contribute $0,
    # Pending rows are excluded since their outcome isn't known yet -- can't total an unknown).
    for d in dates:
        i = row_of[d]
        day_settled = month_df[(month_df['date'] == d) & (month_df['outcome'].isin(['Won', 'Lost', 'Flat']))]
        if day_settled.empty:
            continue
        day_total = day_settled['pnl'].sum()
        z[i][TOTAL_COL] = GRID_CATEGORY['Won' if day_total > 0 else ('Lost' if day_total < 0 else 'Flat')] + 0.5
        text[i][TOTAL_COL] = f"{day_total:+.0f}"
        hover[i][TOTAL_COL] = f"Daily total: {fmt_money(day_total, signed=True)}"

    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(1, 26)), y=date_strs,
        text=text, texttemplate='%{text}', textfont=dict(size=10, color='#eee'),
        customdata=hover, hovertemplate='Date %{y}, Hour %{x}<br>%{customdata}<extra></extra>',
        colorscale=_discrete_category_colorscale(GRID_COLORS), zmin=0, zmax=len(GRID_COLORS),
        showscale=False, xgap=2, ygap=2,
    ))
    fig.add_vline(x=24.5, line_color=COLORS['muted'], line_width=1)
    fig.update_layout(
        template='plotly_dark', title=None,
        xaxis=dict(tickmode='array', tickvals=list(range(1, 25)) + [25], ticktext=[str(h) for h in range(1, 25)] + ['Total'],
                   range=[0.5, 25.5], side='top', gridcolor=COLORS['grid']),
        yaxis=dict(type='category', tickmode='array', tickvals=date_strs, ticktext=date_strs,
                   autorange='reversed'),
        margin=dict(t=30, b=10, l=90, r=10),
        height=40 + n * TABLE_ROW_HEIGHT,
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

        grid_fig = build_case_grid_fig(m_dates, m_df)
        rows_html = '\n'.join(render_trade_row(r) for _, r in m_trades.iterrows())

        sections_html += f'''<div id="month-{m}" class="tab-content {active}">
  <div class="stat-row month-stats">{month_tiles}</div>
  {GRID_LEGEND}
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
<title>Trading Portfolio</title>
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
<h1>Trading Portfolio</h1>
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
</script>
</body>
</html>
"""

    with open('docs/portfolio.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print("Portfolio generated successfully at docs/portfolio.html!")


if __name__ == '__main__':
    build_portfolio()
