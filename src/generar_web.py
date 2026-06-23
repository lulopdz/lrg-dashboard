import pandas as pd
import plotly.graph_objects as go
import os

# 1. Load historical data
df = pd.read_csv('data/ieso_dam_prices.csv', parse_dates=['interval_start_local'])
df = df.sort_values(['location', 'interval_start_local'])

latest_ts = df['interval_start_local'].max()

# IESO hour-ending convention: Hour 1 = 00:00-01:00, ..., Hour 24 = 23:00-00:00
df['hour'] = df['interval_start_local'].dt.hour + 1

zones = sorted(df['location'].unique())
default_zone = 'OTTAWA'
default_idx = zones.index(default_zone)

# 2. Hourly profile: today vs yesterday vs 7-day average, by zone
last_day_rows = df[df['interval_start_local'].dt.date == latest_ts.date()]
if last_day_rows.groupby('location').size().min() < 24:
    # The most recent day is still incomplete: fall back to the prior day
    latest_ts = latest_ts - pd.Timedelta(days=1)

today_date = latest_ts.date()
yesterday_date = today_date - pd.Timedelta(days=1)
week_ago_date = today_date - pd.Timedelta(days=7)

df_today = df[df['interval_start_local'].dt.date == today_date]
df_yesterday = df[df['interval_start_local'].dt.date == yesterday_date]
df_last7 = df[(df['interval_start_local'].dt.date > week_ago_date) & (df['interval_start_local'].dt.date <= today_date)]
avg_7d = df_last7.groupby(['location', 'hour'])['lmp'].mean().reset_index()

hourly_fig = go.Figure()
for i, zone in enumerate(zones):
    visible = (i == default_idx)
    today_z = df_today[df_today['location'] == zone].sort_values('hour')
    yesterday_z = df_yesterday[df_yesterday['location'] == zone].sort_values('hour')
    avg_z = avg_7d[avg_7d['location'] == zone].sort_values('hour')

    hourly_fig.add_trace(go.Scatter(
        x=avg_z['hour'], y=avg_z['lmp'], name='7d Average', mode='lines',
        line=dict(color='#888', dash='dot'), visible=visible, legendgroup=zone
    ))
    hourly_fig.add_trace(go.Scatter(
        x=yesterday_z['hour'], y=yesterday_z['lmp'], name=f'Yesterday ({yesterday_date})', mode='lines+markers',
        line=dict(color='#f1c40f', dash='dash'), visible=visible, legendgroup=zone
    ))
    hourly_fig.add_trace(go.Scatter(
        x=today_z['hour'], y=today_z['lmp'], name=f'Today ({today_date})', mode='lines+markers',
        line=dict(color='#3498db', width=3), visible=visible, legendgroup=zone
    ))

traces_per_zone = 3
hourly_buttons = [
    dict(label=zone, method='update',
         args=[{'visible': [j // traces_per_zone == i for j in range(len(zones) * traces_per_zone)]},
               {'title': f'Hourly Profile - {zone} (Today ({today_date}) vs Yesterday ({yesterday_date}) vs 7d Average)'}])
    for i, zone in enumerate(zones)
]
hourly_fig.update_layout(
    template='plotly_dark',
    title=f'Hourly Profile - {default_zone} (Today ({today_date}) vs Yesterday ({yesterday_date}) vs 7d Average)',
    updatemenus=[dict(buttons=hourly_buttons, direction='down', x=1.0, y=1.18, xanchor='right', yanchor='top',
                       active=default_idx, showactive=True)],
    legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
    xaxis_title='Hour', yaxis_title='Price ($/MWh)',
    xaxis=dict(dtick=1, range=[0.5, 24.5]),
    margin=dict(t=90, b=60, r=140),
    height=500
)

# 3. Hourly price table: dates x hours, colored by price, by zone
TABLE_DAYS = 14
table_start_date = today_date - pd.Timedelta(days=TABLE_DAYS - 1)
df_table = df[(df['interval_start_local'].dt.date >= table_start_date) & (df['interval_start_local'].dt.date <= today_date)].copy()
df_table['date'] = df_table['interval_start_local'].dt.date.astype(str)

table_fig = go.Figure()
for i, zone in enumerate(zones):
    pivot = (df_table[df_table['location'] == zone]
             .pivot_table(index='date', columns='hour', values='lmp', aggfunc='mean')
             .reindex(columns=range(1, 25)))
    text = pivot.round(1).astype(str).values

    table_fig.add_trace(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        text=text, texttemplate='%{text}', textfont=dict(size=10),
        colorscale='RdYlGn_r', colorbar=dict(title='$/MWh'),
        visible=(i == default_idx),
        hovertemplate='Date %{y}, Hour %{x}<br>Price: $%{z:.2f}<extra></extra>'
    ))

table_buttons = [
    dict(label=zone, method='update',
         args=[{'visible': [j == i for j in range(len(zones))]},
               {'title': f'Hourly Price Table - {zone} (last {TABLE_DAYS} days)'}])
    for i, zone in enumerate(zones)
]
table_fig.update_layout(
    template='plotly_dark',
    title=f'Hourly Price Table - {default_zone} (last {TABLE_DAYS} days)',
    updatemenus=[dict(buttons=table_buttons, direction='down', x=1.0, y=1.12, xanchor='right', yanchor='top',
                       active=default_idx, showactive=True)],
    xaxis_title='Hour', yaxis_title='Date',
    xaxis=dict(dtick=1, side='top'),
    margin=dict(t=90, b=40, r=40),
    height=560
)

# 4. Assemble the HTML page
os.makedirs('docs', exist_ok=True)

html = f"""<html>
<head>
<meta charset="utf-8">
<title>DAM Dashboard</title>
<style>
  body {{ background:#111; color:#eee; font-family:Arial, sans-serif; margin:0; padding:24px; }}
  h2 {{ color:#ddd; border-bottom:1px solid #333; padding-bottom:6px; }}
  footer {{ color:#888; font-size:12px; margin-top:20px; }}
</style>
</head>
<body>
<h2>Hourly Profile - Today vs Yesterday vs 7d Average</h2>
{hourly_fig.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>Hourly Price Table</h2>
{table_fig.to_html(full_html=False, include_plotlyjs=False)}
<footer>Last updated: {latest_ts.strftime('%Y-%m-%d %H:%M')}</footer>
</body>
</html>
"""

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Dashboard generated successfully at docs/index.html!")
