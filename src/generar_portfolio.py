"""Builds docs/portfolio.html: the settled trading history from data/historical_pnl.csv
(itself built by parse_reports.py from the IESO participation XML reports in data/reports),
grouped by month, as a per-trade log plus an all-time equity curve and summary stats."""
import os

import pandas as pd
import plotly.graph_objects as go

from dashboard_data import COLORS

os.makedirs('docs', exist_ok=True)

BADGE_CLASS = {'Won': 'badge-won', 'Lost': 'badge-lost', 'Flat': 'badge-flat', 'Pending': 'badge-pending'}


def fmt_money(val, signed=False):
    if pd.isna(val):
        return '—'
    sign = '+' if (signed and val > 0) else ''
    return f"{sign}${val:,.2f}"


def outcome_of(pnl):
    if pd.isna(pnl):
        return 'Pending'
    if pnl > 0:
        return 'Won'
    if pnl < 0:
        return 'Lost'
    return 'Flat'


def load_trades():
    """One row per actually-submitted bid/offer (energy_mw != 0) -- the zero rows in
    historical_pnl.csv just mark hours the resource was eligible for but sat out, which
    isn't a 'trade' to log. position/outcome mirror the vocabulary the Trading Simulator
    already uses (Long/Short on the DAM-RTM spread) so the two pages read as one product."""
    df = pd.read_csv('data/historical_pnl.csv', parse_dates=['date'])
    trades = df[df['energy_mw'] != 0].copy()
    if trades.empty:
        return trades
    trades['position'] = trades['energy_mw'].apply(lambda v: 'Long' if v > 0 else 'Short')
    trades['size'] = trades['energy_mw'].abs()
    trades['outcome'] = trades['pnl'].apply(outcome_of)
    trades['month'] = trades['date'].dt.strftime('%Y-%m')
    return trades.sort_values(['date', 'hour'])


def build_equity_fig(settled):
    """Cumulative PnL across every settled trade, chronological. A single neutral line
    (no series color to assign -- it's the whole portfolio, not a DAM or RTM position) with
    a zero reference and a colored end-marker/value carrying the 'are we up or down' read,
    same treatment as the recommended-hour marker on the forecast charts."""
    x_labels = [f"{d:%b %d} HE{int(h):02d}" for d, h in zip(settled['date'], settled['hour'])]
    pnl_fmt = [fmt_money(v, signed=True) for v in settled['pnl']]
    cum_fmt = [fmt_money(v, signed=True) for v in settled['cum_pnl']]

    fig = go.Figure()
    fig.add_hline(y=0, line_color=COLORS['muted'], line_width=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(settled))), y=settled['cum_pnl'], mode='lines',
        line=dict(color='#d1d5db', width=2),
        customdata=list(zip(x_labels, pnl_fmt, cum_fmt)),
        hovertemplate='%{customdata[0]}<br>Trade PnL: %{customdata[1]}<br>Cumulative: %{customdata[2]}<extra></extra>',
        showlegend=False,
    ))
    final_val = settled['cum_pnl'].iloc[-1]
    end_color = COLORS['positive'] if final_val >= 0 else COLORS['negative']
    fig.add_trace(go.Scatter(
        x=[len(settled) - 1], y=[final_val], mode='markers',
        marker=dict(size=10, color=end_color, line=dict(width=2, color=COLORS['ring'])),
        showlegend=False, hoverinfo='skip',
    ))
    fig.update_layout(
        template='plotly_dark', title=None,
        xaxis=dict(showticklabels=False, gridcolor=COLORS['grid'], zeroline=False, title=None),
        yaxis=dict(gridcolor=COLORS['grid'], title='Cumulative PnL ($)'),
        hovermode='x unified', margin=dict(t=10, b=10, l=60, r=20), height=240,
    )
    return fig


def stat_tile(label, value, cls='', sub=None, extra_cls=''):
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ''
    return f'''<div class="stat-tile {extra_cls}"><div class="stat-label">{label}</div>
    <div class="stat-value {cls}">{value}</div>{sub_html}</div>'''


def pnl_cls(val):
    if pd.isna(val) or val == 0:
        return ''
    return 'pos' if val > 0 else 'neg'


def render_trade_row(r):
    pos_cls = 'badge-long' if r['position'] == 'Long' else 'badge-short'
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
    trades = load_trades()
    if trades.empty:
        print("No trades found in historical_pnl.csv (run parse_reports.py first).")
        return

    settled = trades[trades['outcome'] != 'Pending'].copy().sort_values(['date', 'hour'])
    settled['cum_pnl'] = settled['pnl'].cumsum()

    n_trades, n_settled = len(trades), len(settled)
    n_won = int((settled['pnl'] > 0).sum())
    n_lost = int((settled['pnl'] < 0).sum())
    win_rate = n_won / n_settled if n_settled else 0
    net_pnl = settled['pnl'].sum() if n_settled else 0
    avg_pnl = settled['pnl'].mean() if n_settled else 0
    pending_sub = f"{n_trades - n_settled} pending settlement" if n_trades > n_settled else None

    overview_tiles = stat_tile('Total trades', n_trades, sub=pending_sub)
    overview_tiles += stat_tile('Win rate', f"{win_rate:.0%}", sub=f"{n_won} won / {n_lost} lost")
    overview_tiles += stat_tile('Net PnL', fmt_money(net_pnl, signed=True), cls=pnl_cls(net_pnl), extra_cls='highlight')
    overview_tiles += stat_tile('Avg PnL / trade', fmt_money(avg_pnl, signed=True), cls=pnl_cls(avg_pnl))
    if n_settled:
        best, worst = settled.loc[settled['pnl'].idxmax()], settled.loc[settled['pnl'].idxmin()]
        overview_tiles += stat_tile('Best trade', fmt_money(best['pnl'], signed=True), cls='pos',
                                     sub=f"{best['date']:%b %d} HE{int(best['hour']):02d}")
        overview_tiles += stat_tile('Worst trade', fmt_money(worst['pnl'], signed=True), cls='neg',
                                     sub=f"{worst['date']:%b %d} HE{int(worst['hour']):02d}")

    if n_settled:
        equity_fig = build_equity_fig(settled)
        equity_html = f'''<div class="stat-label" style="margin:20px 0 8px;">Cumulative PnL
      &middot; {settled['date'].min():%b %d, %Y} &rarr; {settled['date'].max():%b %d, %Y}</div>
    {equity_fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='equity-curve')}'''
    else:
        equity_html = ''

    months = sorted(trades['month'].unique(), reverse=True)
    tabs_html, sections_html = '', ''
    for i, m in enumerate(months):
        active = 'active' if i == 0 else ''
        month_label = pd.to_datetime(m + '-01').strftime('%B %Y')
        tabs_html += f'<button class="tab-btn {active}" onclick="showMonth(\'{m}\', this)">{month_label}</button>\n'

        m_trades = trades[trades['month'] == m].sort_values(['date', 'hour'], ascending=[False, True])
        m_settled = m_trades[m_trades['outcome'] != 'Pending']
        m_won, m_lost = int((m_settled['pnl'] > 0).sum()), int((m_settled['pnl'] < 0).sum())
        m_net = m_settled['pnl'].sum() if len(m_settled) else 0

        month_tiles = stat_tile('Trades', len(m_trades))
        month_tiles += stat_tile('Won / Lost', f'<span class="pos">{m_won}</span> / <span class="neg">{m_lost}</span>')
        month_tiles += stat_tile('Net PnL', fmt_money(m_net, signed=True), cls=pnl_cls(m_net))
        month_tiles += stat_tile('Active days', m_trades['date'].nunique())

        rows_html = '\n'.join(render_trade_row(r) for _, r in m_trades.iterrows())

        sections_html += f'''<div id="month-{m}" class="tab-content {active}">
  <div class="stat-row month-stats">{month_tiles}</div>
  <div class="table-container">
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
  h2 {{ font-size:15px; color:#ccc; margin:28px 0 12px; }}
  .subtitle {{ color:#888; font-size:13px; margin:0 0 20px; max-width:760px; }}
  .back-link {{ display:inline-block; margin-bottom:16px; color:#aaa; text-decoration:none; font-size:13px; }}
  .back-link:hover {{ color:#fff; }}

  .card {{ background:#161616; border:1px solid #2a2a2a; border-radius:8px; padding:16px 20px; margin:16px 0; }}

  .stat-row {{ display:flex; flex-wrap:wrap; gap:16px; margin:0; }}
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

  .tab-content {{ display:none; }}
  .tab-content.active {{ display:block; }}

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
<p class="subtitle">Settled results for every LRG bid/offer submitted to the IESO market, built from the
participation reports in data/reports. Long bets the DAM price clears above RTM; Short bets the opposite.
Positive PnL means the trade closed in your favor.</p>

<div class="card">
  <div class="stat-row">{overview_tiles}</div>
  {equity_html}
</div>

<h2>Monthly detail</h2>
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
