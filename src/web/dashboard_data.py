"""Loads IESO/Open-Meteo data and computes the shared config, zones, variable maps, and
day options that both dashboard_figures.py and generar_web.py build on.

Importing this module reads every price/weather CSV (~20MB). Anything that only needs the
palette or figure sizes should import theme.py instead, which has no data dependency."""
import os

import pandas as pd

from theme import COLORS  # noqa: F401 -- re-exported: existing importers read COLORS from here

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

# How much to trust the forecast above -- ensemble percentiles and run-to-run revisions,
# written by update_weather_confidence.py. Optional: it only started being collected recently
# and the ensemble endpoint serves a short window, so it covers a handful of days around now
# rather than the full weather history. Everything downstream treats it as best-effort.
_confidence_path = 'data/weather_confidence.csv'
if os.path.exists(_confidence_path):
    weather_confidence = pd.read_csv(_confidence_path, parse_dates=['timestamp'])
    weather_confidence = weather_confidence.sort_values('timestamp')
    weather_confidence['hour'] = weather_confidence['timestamp'].dt.hour + 1
else:
    weather_confidence = None

load_forecast = pd.read_csv('data/ieso_load_forecast.csv', parse_dates=['interval_start_local'])
load_forecast = load_forecast.sort_values('interval_start_local')
load_forecast['hour'] = load_forecast['interval_start_local'].dt.hour + 1

# IESO's adequacy report: scheduled generation by fuel, outages, interchange and reserve.
# Optional like weather_confidence -- update_adequacy.py is newer than the other feeds, so a
# checkout without it should still build the rest of the page.
_adequacy_path = 'data/ieso_adequacy.csv'
if os.path.exists(_adequacy_path):
    adequacy = pd.read_csv(_adequacy_path, parse_dates=['interval_start_local'])
    adequacy = adequacy.sort_values('interval_start_local')
    adequacy['hour'] = adequacy['interval_start_local'].dt.hour + 1
else:
    adequacy = None

# Fuel -> display label for the supply-mix chart, ordered the way they stack: the big,
# steady baseload at the bottom and the small intermittent ones on top, so the visually
# noisy series don't shift everything above them up and down hour to hour.
SUPPLY_MIX = {
    'nuclear_scheduled': 'Nuclear',
    'hydro_scheduled': 'Hydro',
    'gas_scheduled': 'Gas',
    'wind_scheduled': 'Wind',
    'solar_scheduled': 'Solar',
    'biofuel_scheduled': 'Biofuel',
}

wind_forecast = pd.read_csv('data/ieso_wind_forecast.csv', parse_dates=['interval_start_local'])
wind_forecast = wind_forecast.sort_values(['zone', 'interval_start_local'])
wind_forecast['hour'] = wind_forecast['interval_start_local'].dt.hour + 1

zones = sorted(dam['location'].unique())
default_idx = zones.index(DEFAULT_ZONE)

# Order is the display order of the Weather tab's small-multiples grid (3 across), so the
# first three fill the top row: temperature, wind at the turbine site, humidity. Port Alma
# (42.18N, -82.24W, Chatham-Kent) is where the wind fleet sits, so it's the wind speed that
# moves price; Ottawa's is the pricing zone's own weather. The two sit in the same grid
# column (positions 2 and 5) so they stack vertically and can be read against each other.
WEATHER_VARS = {
    # Toronto -- the province's load centre, so its weather is a demand signal. Its wind and
    # humidity correlate only ~0.45-0.48 with Ottawa's, i.e. they carry their own information
    # rather than restating it; temperature tracks closely (0.96) but runs ~2 C warmer.
    "temperature_2m_toronto": ("Temperature", "°C"),
    "wind_speed_10m_toronto": ("Wind 10m", "m/s"),
    "relative_humidity_2m_toronto": ("Humidity", "%"),
    # Ottawa -- the pricing zone itself.
    "temperature_2m": ("Temperature", "°C"),
    "wind_speed_10m": ("Wind 10m", "m/s"),
    "relative_humidity_2m": ("Humidity", "%"),
    # Port Alma -- where the wind fleet sits, so these are supply signals.
    "wind_speed_100m_port_alma": ("Wind 100m", "m/s"),
    "shortwave_radiation_port_alma": ("Solar Radiation", "W/m²"),
    # Remaining Ottawa conditions.
    "shortwave_radiation": ("Solar Radiation", "W/m²"),
    "precipitation": ("Precipitation", "mm"),
    "snowfall": ("Snowfall", "cm"),
}

# Section -> variables, in render order. Labels above are deliberately bare (no site name)
# because the section heading carries it; that also keeps the three-column grid aligned, so
# Toronto's temperature sits directly above Ottawa's and the two read as one comparison.
WEATHER_GROUPS = {
    "Demand centre · Toronto": ["temperature_2m_toronto", "wind_speed_10m_toronto",
                                 "relative_humidity_2m_toronto"],
    "Pricing zone · Ottawa": ["temperature_2m", "wind_speed_10m", "relative_humidity_2m"],
    "Generation site · Port Alma": ["wind_speed_100m_port_alma", "shortwave_radiation_port_alma"],
    "Other conditions · Ottawa": ["shortwave_radiation", "precipitation", "snowfall"],
}

# Where a label appears away from its section heading (stat tiles, the revision table, the
# hourly table's variable picker) it needs the site spelled out -- including Ottawa's, which
# would otherwise be the only unlabelled ones and read as if the site were missing rather
# than implied.
WEATHER_SITE = {
    "temperature_2m_toronto": "Toronto", "wind_speed_10m_toronto": "Toronto",
    "relative_humidity_2m_toronto": "Toronto",
    "wind_speed_100m_port_alma": "Port Alma", "shortwave_radiation_port_alma": "Port Alma",
    "temperature_2m": "Ottawa", "wind_speed_10m": "Ottawa", "relative_humidity_2m": "Ottawa",
    "shortwave_radiation": "Ottawa", "precipitation": "Ottawa", "snowfall": "Ottawa",
}


def weather_label(key, qualified=True):
    """Display label for a weather variable, optionally with its site."""
    label = WEATHER_VARS[key][0]
    site = WEATHER_SITE.get(key)
    return f"{label} · {site}" if qualified and site else label


def weather_label_html(key):
    """Same, with the site as a dimmer suffix so the variable name stays the thing you scan
    and the qualifier doesn't widen the column as much as plain text would."""
    label = WEATHER_VARS[key][0]
    site = WEATHER_SITE.get(key)
    return f'{label} <span class="site">{site}</span>' if site else label
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

# Dates selectable in the shared "Day" picker: same window, plus tomorrow (DAM is already
# published for it). Every tab that plots by day -- DAM/RTM/Spread/Weather/Load/Wind --
# reads off this one list, but the two tab groups default to different entries: the market
# tabs (DAM/RTM/Spread) open on today, the forecast tabs (Weather/Load/Wind, which all carry
# a look-ahead forecast) open on tomorrow.
DAY_OPTIONS = SELECTABLE_DATES + [today_date + pd.Timedelta(days=1)]
DAY_OPTION_STRS = [str(d) for d in DAY_OPTIONS]
default_date_idx = len(DAY_OPTIONS) - 2
default_forecast_date_idx = len(DAY_OPTIONS) - 1
