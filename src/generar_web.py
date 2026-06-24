import pandas as pd
import plotly.graph_objects as go
import os

TABLE_DAYS = 14
DEFAULT_ZONE = 'OTTAWA'

# GitHub repo that hosts this dashboard, used to build the link the "Refresh
# RTM" button opens (the GitHub Actions page for the refresh_rtm workflow).
GITHUB_OWNER = 'lulopdz'
GITHUB_REPO = 'lrg-dashboard'

# 1. Load DAM and RTM (both stored hourly; update_rtm.py aggregates the raw 5-min feed)
dam = pd.read_csv('data/ieso_dam_prices.csv', parse_dates=['interval_start_local'])
dam = dam.sort_values(['location', 'interval_start_local'])
dam['hour'] = dam['interval_start_local'].dt.hour + 1  # IESO hour-ending convention: 1-24

rtm = pd.read_csv('data/ieso_rtm_prices.csv', parse_dates=['interval_start_local'])
rtm = rtm.sort_values(['location', 'interval_start_local'])
rtm['hour'] = rtm['interval_start_local'].dt.hour + 1

zones = sorted(dam['location'].unique())
default_idx = zones.index(DEFAULT_ZONE)

# 2. Spread = DAM - RTM, aligned by zone/hour
spread = dam[['location', 'interval_start_local', 'hour', 'lmp']].merge(
    rtm[['location', 'interval_start_local', 'hour', 'lmp']],
    on=['location', 'interval_start_local', 'hour'], suffixes=('_dam', '_rtm')
)
spread['lmp'] = spread['lmp_dam'] - spread['lmp_rtm']
spread = spread[['location', 'interval_start_local', 'hour', 'lmp']]

# 3. Shared reference dates, driven by RTM since it always lags DAM (which publishes a day ahead)
latest_ts = rtm['interval_start_local'].max()
last_day_rows = rtm[rtm['interval_start_local'].dt.date == latest_ts.date()]
if last_day_rows.groupby('location').size().min() < 24:
    # The most recent day is still incomplete: fall back to the prior day
    latest_ts = latest_ts - pd.Timedelta(days=1)

today_date = latest_ts.date()
yesterday_date = today_date - pd.Timedelta(days=1)
week_ago_date = today_date - pd.Timedelta(days=7)
table_start_date = today_date - pd.Timedelta(days=TABLE_DAYS - 1)


def build_hourly_fig(df, label):
    df_today = df[df['interval_start_local'].dt.date == today_date]
    df_yesterday = df[df['interval_start_local'].dt.date == yesterday_date]
    df_last7 = df[(df['interval_start_local'].dt.date > week_ago_date) & (df['interval_start_local'].dt.date <= today_date)]
    avg_7d = df_last7.groupby(['location', 'hour'])['lmp'].mean().reset_index()

    fig = go.Figure()
    for i, zone in enumerate(zones):
        visible = (i == default_idx)
        today_z = df_today[df_today['location'] == zone].sort_values('hour')
        yesterday_z = df_yesterday[df_yesterday['location'] == zone].sort_values('hour')
        avg_z = avg_7d[avg_7d['location'] == zone].sort_values('hour')

        fig.add_trace(go.Scatter(
            x=avg_z['hour'], y=avg_z['lmp'], name='7d Average', mode='lines',
            line=dict(color='#888', dash='dot'), visible=visible, legendgroup=zone
        ))
        fig.add_trace(go.Scatter(
            x=yesterday_z['hour'], y=yesterday_z['lmp'], name=f'{yesterday_date}', mode='lines+markers',
            line=dict(color='#f1c40f', dash='dash'), visible=visible, legendgroup=zone
        ))
        fig.add_trace(go.Scatter(
            x=today_z['hour'], y=today_z['lmp'], name=f'{today_date}', mode='lines+markers',
            line=dict(color='#3498db', width=3), visible=visible, legendgroup=zone
        ))

    traces_per_zone = 3
    buttons = [
        dict(label=zone, method='update',
             args=[{'visible': [j // traces_per_zone == i for j in range(len(zones) * traces_per_zone)]},
                   {'title': f'{label} - Hourly Profile - {zone}'}])
        for i, zone in enumerate(zones)
    ]
    fig.update_layout(
        template='plotly_dark',
        title=f'{label} - Hourly Profile - {DEFAULT_ZONE}',
        updatemenus=[dict(buttons=buttons, direction='down', x=1.0, y=1.18, xanchor='right', yanchor='top',
                           active=default_idx, showactive=True)],
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
        xaxis_title='Hour', yaxis_title='Price ($/MWh)',
        xaxis=dict(dtick=1, range=[0.5, 24.5]),
        margin=dict(t=90, b=60, r=140),
        height=500
    )
    return fig


def build_table_fig(df, label, diverging=False):
    df_table = df[(df['interval_start_local'].dt.date >= table_start_date) & (df['interval_start_local'].dt.date <= today_date)].copy()
    df_table['date'] = df_table['interval_start_local'].dt.date.astype(str)

    heatmap_kwargs = dict(colorscale='RdYlGn', zmid=0) if diverging else dict(colorscale='RdYlGn_r')
    all_dates = [str(table_start_date + pd.Timedelta(days=d)) for d in range(TABLE_DAYS)]

    fig = go.Figure()
    for i, zone in enumerate(zones):
        pivot = (df_table[df_table['location'] == zone]
                 .pivot_table(index='date', columns='hour', values='lmp', aggfunc='mean')
                 .reindex(index=all_dates, columns=range(1, 25)))
        text = pivot.round(1).astype(str).values

        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            text=text, texttemplate='%{text}', textfont=dict(size=10),
            colorbar=dict(title='$/MWh'),
            visible=(i == default_idx),
            hovertemplate='Date %{y}, Hour %{x}<br>Price: $%{z:.2f}<extra></extra>',
            **heatmap_kwargs
        ))

    buttons = [
        dict(label=zone, method='update',
             args=[{'visible': [j == i for j in range(len(zones))]},
                   {'title': f'{label} - Hourly Table - {zone} (last {TABLE_DAYS} days)'}])
        for i, zone in enumerate(zones)
    ]
    fig.update_layout(
        template='plotly_dark',
        title=f'{label} - Hourly Table - {DEFAULT_ZONE} (last {TABLE_DAYS} days)',
        updatemenus=[dict(buttons=buttons, direction='down', x=1.0, y=1.12, xanchor='right', yanchor='top',
                           active=default_idx, showactive=True)],
        xaxis_title='Hour', yaxis_title='Date',
        xaxis=dict(dtick=1, side='top'),
        yaxis=dict(tickmode='array', tickvals=all_dates, ticktext=all_dates),
        margin=dict(t=90, b=40, r=40),
        height=560
    )
    return fig


def build_spread_sign_fig(df):
    df_today = df[df['interval_start_local'].dt.date == today_date]

    fig = go.Figure()
    for i, zone in enumerate(zones):
        z = df_today[df_today['location'] == zone].sort_values('hour')
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in z['lmp']]
        fig.add_trace(go.Bar(
            x=z['hour'], y=z['lmp'], marker_color=colors,
            visible=(i == default_idx), showlegend=False,
            hovertemplate='Hour %{x}<br>Spread: $%{y:.2f}<extra></extra>'
        ))

    buttons = [
        dict(label=zone, method='update',
             args=[{'visible': [j == i for j in range(len(zones))]},
                   {'title': f'Spread Sign - {zone} ({today_date})'}])
        for i, zone in enumerate(zones)
    ]
    fig.update_layout(
        template='plotly_dark',
        title=f'Spread Sign - {DEFAULT_ZONE} ({today_date})',
        updatemenus=[dict(buttons=buttons, direction='down', x=1.0, y=1.18, xanchor='right', yanchor='top',
                           active=default_idx, showactive=True)],
        xaxis_title='Hour', yaxis_title='Spread ($/MWh)',
        xaxis=dict(dtick=1, range=[0.5, 24.5]),
        margin=dict(t=90, b=40, r=40),
        height=400
    )
    fig.add_hline(y=0, line_color='#666', line_width=1)
    return fig


dam_hourly_fig = build_hourly_fig(dam, 'DAM')
dam_table_fig = build_table_fig(dam, 'DAM')

rtm_hourly_fig = build_hourly_fig(rtm, 'RTM')
rtm_table_fig = build_table_fig(rtm, 'RTM')

spread_sign_fig = build_spread_sign_fig(spread)
spread_hourly_fig = build_hourly_fig(spread, 'Spread (DAM - RTM)')
spread_table_fig = build_table_fig(spread, 'Spread (DAM - RTM)', diverging=True)

# 4. Assemble the HTML page with tabs (DAM / RTM / Spread)
os.makedirs('docs', exist_ok=True)

html = f"""<html>
<head>
<meta charset="utf-8">
<title>DAM Dashboard</title>
<style>
  body {{ background:#111; color:#eee; font-family:Arial, sans-serif; margin:0; padding:24px; }}
  h2 {{ color:#ddd; border-bottom:1px solid #333; padding-bottom:6px; }}
  footer {{ color:#888; font-size:12px; margin-top:20px; }}
  .tabs {{ display:flex; gap:8px; margin-bottom:16px; }}
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
  /* max-height:0 (instead of display:none) keeps the container's width intact so
     Plotly's auto-sizing doesn't collapse hidden charts to zero width on first render */
  .tab-content {{ max-height:0; overflow:hidden; }}
  .tab-content.active {{ max-height:none; }}
</style>
</head>
<body>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('dam', this)">Day-Ahead Market</button>
  <button class="tab-btn" onclick="showTab('rtm', this)">Real-Time Market</button>
  <button class="tab-btn" onclick="showTab('spread', this)">Spread</button>
</div>

<div id="tab-dam" class="tab-content active">
<h2>DAM - Hourly Profile</h2>
{dam_hourly_fig.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>DAM - Hourly Table</h2>
{dam_table_fig.to_html(full_html=False, include_plotlyjs=False)}
</div>

<div id="tab-rtm" class="tab-content">
<div>
  <a class="refresh-btn" href="https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/refresh_rtm.yml" target="_blank" rel="noopener">
    Refresh RTM (opens GitHub Actions)
  </a>
</div>
<h2>RTM - Hourly Profile</h2>
{rtm_hourly_fig.to_html(full_html=False, include_plotlyjs=False)}
<h2>RTM - Hourly Table</h2>
{rtm_table_fig.to_html(full_html=False, include_plotlyjs=False)}
</div>

<div id="tab-spread" class="tab-content">
<h2>Spread Sign - Today (Positive = green, Negative = red)</h2>
{spread_sign_fig.to_html(full_html=False, include_plotlyjs=False)}
<h2>Spread (DAM - RTM) - Hourly Profile</h2>
{spread_hourly_fig.to_html(full_html=False, include_plotlyjs=False)}
<h2>Spread (DAM - RTM) - Hourly Table</h2>
{spread_table_fig.to_html(full_html=False, include_plotlyjs=False)}
</div>

<footer>Last updated: {latest_ts.strftime('%Y-%m-%d %H:%M')}</footer>

<script>
function showTab(name, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}}
</script>

</body>
</html>
"""

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Dashboard generated successfully at docs/index.html!")
