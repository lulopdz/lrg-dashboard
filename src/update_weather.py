import math
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

OTTAWA_COORDS = {"lat": 45.4000, "lon": -75.7000}
PORT_ALMA_COORDS = {"lat": 42.1808, "lon": -82.2444}  # secondary wind-speed station

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "shortwave_radiation",
    "global_tilted_irradiance",
]

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "OTTAWA_weather.csv"
CACHE_PATH = Path(__file__).resolve().parent.parent / ".cache"
DEFAULT_PAST_DAYS = 7  # used only when there's no existing file to anchor from
FORECAST_DAYS = 3


def _client():
    cache_session = requests_cache.CachedSession(str(CACHE_PATH), expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _fetch(openmeteo, coords, variables, past_days, forecast_days):
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "hourly": variables,
        "timezone": "UTC",  # always fetch UTC, convert locally
        "past_days": past_days,
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",
    }
    response = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
    hourly = response.Hourly()
    start_time = pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("-05:00")
    timestamps = pd.date_range(
        start=start_time,
        periods=len(hourly.Variables(0).ValuesAsNumpy()),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    data = {var: hourly.Variables(i).ValuesAsNumpy() for i, var in enumerate(variables)}
    return pd.DataFrame(data, index=timestamps)


def determine_past_days():
    if not DATA_PATH.exists():
        return DEFAULT_PAST_DAYS
    existing = pd.read_csv(DATA_PATH, usecols=["timestamp"])
    if existing.empty:
        return DEFAULT_PAST_DAYS
    last_ts = pd.to_datetime(existing["timestamp"], errors="coerce").max()
    now = pd.Timestamp.now(tz="UTC").tz_convert("-05:00")
    hours_old = (now - last_ts).total_seconds() / 3600
    return max(1, math.ceil(hours_old / 24) + 1)


def update_weather():
    past_days = determine_past_days()
    print(f"Fetching OTTAWA weather: past_days={past_days}, forecast_days={FORECAST_DAYS}...")

    openmeteo = _client()
    df_new = _fetch(openmeteo, OTTAWA_COORDS, WEATHER_VARIABLES, past_days, FORECAST_DAYS)
    df_new.index.name = "timestamp"
    df_new = df_new.reset_index()

    port_alma = _fetch(openmeteo, PORT_ALMA_COORDS, ["wind_speed_10m"], past_days, FORECAST_DAYS)
    port_alma = port_alma.rename(columns={"wind_speed_10m": "wind_speed_10m_port_alma"})
    df_new = df_new.set_index("timestamp").join(port_alma, how="left").reset_index()

    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH)
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], errors="coerce")
        combined = pd.concat([existing_df, df_new], axis=0)
    else:
        combined = df_new

    # Drop duplicates keeping the LAST one (newest forecast), same convention as the
    # price feeds: as the forecast horizon shrinks, fresher predictions overwrite older ones.
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp", ascending=False)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(combined)} rows to {DATA_PATH} (latest: {combined['timestamp'].max()})")


if __name__ == "__main__":
    update_weather()
