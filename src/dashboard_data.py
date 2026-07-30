"""Loads IESO/Open-Meteo data and computes the shared config, zones, variable maps, and
day options that both dashboard_figures.py and generar_web.py build on."""
import pandas as pd

TABLE_DAYS = 30
DEFAULT_ZONE = 'OTTAWA'

# GitHub repo that hosts this dashboard, used to build the links the "Refresh"
# buttons open (the GitHub Actions pages for each workflow).
GITHUB_OWNER = 'lulopdz'
GITHUB_REPO = 'lrg-dashboard'

# 1. Load DAM and RTM (both stored hourly; update_rtm.py aggregates the raw 5-min feed)
dam = pd.read_csv('data/ieso_dam_prices.csv', parse_dates=['interval_start_local'])
dam = dam.sort_values(['location', 'interval_start_local'])
dam['hour'] = dam['interval_start_local'].dt.hour + 1  # IESO hour-ending convention: 1-24

rtm = pd.read_csv('data/ieso_rtm_prices.csv', parse_dates=['interval_start_local'])
rtm = rtm.sort_values(['location', 'interval_start_local'])
rtm['hour'] = rtm['interval_start_local'].dt.hour + 1

weather = pd.read_csv('data/OTTAWA_weather.csv', parse_dates=['timestamp'])
weather = weather.sort_values('timestamp')
weather['hour'] = weather['timestamp'].dt.hour + 1

load_forecast = pd.read_csv('data/ieso_load_forecast.csv', parse_dates=['interval_start_local'])
load_forecast = load_forecast.sort_values('interval_start_local')
load_forecast['hour'] = load_forecast['interval_start_local'].dt.hour + 1

wind_forecast = pd.read_csv('data/ieso_wind_forecast.csv', parse_dates=['interval_start_local'])
wind_forecast = wind_forecast.sort_values(['zone', 'interval_start_local'])
wind_forecast['hour'] = wind_forecast['interval_start_local'].dt.hour + 1

zones = sorted(dam['location'].unique())
default_idx = zones.index(DEFAULT_ZONE)

WEATHER_VARS = {
    "temperature_2m": ("Temperature", "°C"),
    "wind_speed_10m": ("Wind Speed", "m/s"),
    "precipitation": ("Precipitation", "mm"),
    "snowfall": ("Snowfall", "cm"),
    "shortwave_radiation": ("Solar Radiation", "W/m²"),
    "relative_humidity_2m": ("Humidity", "%"),
}
weather_var_keys = list(WEATHER_VARS.keys())
default_weather_idx = 0  # Temperature

LOAD_VARS = {
    "ontario": ("Ontario (Total)", "MW"),
    "ontario_northeast": ("Northeast", "MW"),
    "ontario_northwest": ("Northwest", "MW"),
    "ontario_southwest": ("Southwest", "MW"),
    "ontario_southeast": ("Southeast", "MW"),
}
load_var_keys = list(LOAD_VARS.keys())
default_load_idx = 0  # Ontario (Total)

wind_zones = sorted(wind_forecast['zone'].unique())
default_wind_idx = wind_zones.index('Ontario Total') if 'Ontario Total' in wind_zones else 0

# 2. Spread = DAM - RTM, aligned by zone/hour
spread = dam[['location', 'interval_start_local', 'hour', 'lmp']].merge(
    rtm[['location', 'interval_start_local', 'hour', 'lmp']],
    on=['location', 'interval_start_local', 'hour'], suffixes=('_dam', '_rtm')
)
spread['lmp'] = spread['lmp_dam'] - spread['lmp_rtm']
spread = spread[['location', 'interval_start_local', 'hour', 'lmp']]

# 3. Shared reference date: the actual calendar day, in market time. DAM always publishes
# a day ahead (so its max date is "tomorrow", not "today"), and RTM is only ever as
# complete as "right now" -- neither dataset's max date is the right anchor. Using the
# real wall-clock date keeps DAM/RTM/Spread all defaulting to the same meaningful day.
today_date = pd.Timestamp.now(tz='UTC').tz_convert('-05:00').date()
table_start_date = today_date - pd.Timedelta(days=TABLE_DAYS - 1)
latest_ts = dam['interval_start_local'].max()
rtm_latest_ts = rtm['interval_start_local'].max()
weather_latest_ts = weather['timestamp'].max()
load_latest_ts = load_forecast['interval_start_local'].max()
wind_latest_ts = wind_forecast['interval_start_local'].max()

# Dates used by the hourly tables: the rolling last TABLE_DAYS ending at today (unaffected by the Day selector)
SELECTABLE_DATES = [table_start_date + pd.Timedelta(days=d) for d in range(TABLE_DAYS)]
SELECTABLE_DATE_STRS = [str(d) for d in SELECTABLE_DATES]

# Dates selectable in the "Day" dropdown: same window, plus tomorrow (DAM is already
# published for it), defaulting to today rather than the last (tomorrow) entry.
DAY_OPTIONS = SELECTABLE_DATES + [today_date + pd.Timedelta(days=1)]
DAY_OPTION_STRS = [str(d) for d in DAY_OPTIONS]
default_date_idx = len(DAY_OPTIONS) - 2
default_weather_date_idx = len(DAY_OPTIONS) - 1  # tomorrow -- weather forecasts are most useful looking ahead
