"""Shared plumbing for the next-day DAM/RTM predictors (predict_dam.py, predict_rtm.py):
loading IESO's own forecasts, calendar feature engineering, backtesting, and the
similar-day analog search. Both predictors need this unchanged, so it lives here once."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ZONE = "OTTAWA"
WIND_ZONE = "Ontario Total"
BACKTEST_DAYS = 21   # trailing holdout window used to report honest accuracy
MIN_TRAIN_DAYS = 60  # need enough history before the holdout window to bother training

# The full set of forecasted variables we have for tomorrow (IESO load forecast +
# IESO wind forecast + Open-Meteo weather forecast) -- "similar day" means closest
# on these, not on anything realized/actual.
ANALOG_FEATURE_COLS = [
    "ontario", "ontario_southeast", "wind_forecast",
    "temperature_2m", "wind_speed_10m", "precipitation", "snowfall",
    "relative_humidity_2m", "shortwave_radiation",
]
ANALOG_FEATURE_LABELS = {
    "ontario": ("Ontario load forecast", "MW"),
    "ontario_southeast": ("SE Ontario load forecast", "MW"),
    "wind_forecast": ("Wind generation forecast", "MW"),
    "temperature_2m": ("Temperature", "°C"),
    "wind_speed_10m": ("Wind speed", "m/s"),
    "precipitation": ("Precipitation", "mm"),
    "snowfall": ("Snowfall", "cm"),
    "relative_humidity_2m": ("Humidity", "%"),
    "shortwave_radiation": ("Solar radiation", "W/m²"),
}


def load_price_series(filename, zone=ZONE):
    """DAM or RTM hourly price for one zone -- both CSVs share the same
    interval_start_local / location / lmp schema."""
    df = pd.read_csv(DATA_DIR / filename, parse_dates=["interval_start_local"])
    df = df[df["location"] == zone][["interval_start_local", "lmp"]].sort_values("interval_start_local")
    return df


def load_forecast_inputs():
    """The exogenous forecasts available for tomorrow, common to both predictors."""
    load_cols = ["ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast"]
    load_fc = pd.read_csv(DATA_DIR / "ieso_load_forecast.csv", parse_dates=["interval_start_local"])
    load_fc = load_fc[["interval_start_local"] + load_cols].drop_duplicates("interval_start_local", keep="last")

    wind_fc = pd.read_csv(DATA_DIR / "ieso_wind_forecast.csv", parse_dates=["interval_start_local"])
    wind_fc = wind_fc[wind_fc["zone"] == WIND_ZONE][["interval_start_local", "generation_forecast"]]
    wind_fc = wind_fc.rename(columns={"generation_forecast": "wind_forecast"}).drop_duplicates("interval_start_local", keep="last")

    weather = pd.read_csv(DATA_DIR / "OTTAWA_weather.csv", parse_dates=["timestamp"])
    weather = weather.rename(columns={"timestamp": "interval_start_local"}).drop_duplicates("interval_start_local", keep="last")

    return load_fc, wind_fc, weather


def determine_target_date(dam):
    """Tomorrow, anchored on DAM (published as a full day at once) so the DAM and RTM
    predictors always target the same day -- otherwise a predicted spread wouldn't line up."""
    return (dam["interval_start_local"].max() + pd.Timedelta(days=1)).normalize().date()


def build_grid(target_df, load_fc, wind_fc, weather, target_date, tz):
    """One continuous hourly grid spanning history through target_date, with the target
    series ('lmp'), forecast inputs, and calendar features merged in. 'lmp' is NaN for
    target_date -- that's the row(s) to predict."""
    start = target_df["interval_start_local"].min()
    end = pd.Timestamp(target_date, tz=tz) + pd.Timedelta(hours=23)
    idx = pd.date_range(start, end, freq="h", tz=tz)

    df = pd.DataFrame({"interval_start_local": idx})
    df = df.merge(target_df, on="interval_start_local", how="left")
    df = df.merge(load_fc, on="interval_start_local", how="left")
    df = df.merge(wind_fc, on="interval_start_local", how="left")
    df = df.merge(weather, on="interval_start_local", how="left")

    df["hour"] = df["interval_start_local"].dt.hour
    df["dow"] = df["interval_start_local"].dt.dayofweek
    df["month"] = df["interval_start_local"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    for col, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    return df


def add_lag_features(df, prefix):
    """Causal same-hour lags/rolling means on df['lmp'], shifted by whole days on the
    complete hourly grid so nothing at hour t ever sees data from t or later."""
    lmp = df["lmp"]
    df[f"{prefix}_lag_1d"] = lmp.shift(24)
    df[f"{prefix}_lag_7d"] = lmp.shift(24 * 7)
    df[f"{prefix}_roll_7d"] = lmp.shift(24).rolling(24 * 7, min_periods=24 * 3).mean()
    df[f"{prefix}_roll_28d"] = lmp.shift(24).rolling(24 * 28, min_periods=24 * 7).mean()
    return df


def backtest(df_hist, feature_cols, naive_col):
    """Trailing holdout: honest accuracy vs. a naive 'same hour last week' baseline, plus
    a per-hour-of-day error breakdown (which hours the model has historically nailed vs.
    missed) used to recommend the hour we're most confident in."""
    cutoff = df_hist["interval_start_local"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    train = df_hist[df_hist["interval_start_local"] < cutoff]
    test = df_hist[df_hist["interval_start_local"] >= cutoff]
    if len(train) < MIN_TRAIN_DAYS * 24 or test.empty:
        return None

    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(train[feature_cols], train["lmp"])
    pred = model.predict(test[feature_cols])

    naive = test[naive_col].fillna(train["lmp"].mean())

    abs_err = np.abs(test["lmp"].values - pred)
    hourly_mae = (
        pd.DataFrame({"hour": test["hour"].values + 1, "abs_err": abs_err})
        .groupby("hour")["abs_err"].mean()
        .reindex(range(1, 25))
    )

    return {
        "n_test_hours": int(len(test)),
        "model_mae": float(mean_absolute_error(test["lmp"], pred)),
        "model_rmse": float(np.sqrt(mean_squared_error(test["lmp"], pred))),
        "naive_mae": float(mean_absolute_error(test["lmp"], naive)),
        "hourly_mae": {int(h): (None if pd.isna(v) else float(v)) for h, v in hourly_mae.items()},
    }


def recommend_hour(metrics, df_target, feature_cols):
    """The hour (1-24) with the lowest historical backtest error, restricted to hours
    whose target-day inputs are actually complete -- what we surface as 'most confident
    hour'. Returns (hour, expected_mae) or (None, None) if there's nothing to go on."""
    if not metrics or not metrics.get("hourly_mae"):
        return None, None

    missing_mask = df_target[feature_cols].isna().any(axis=1)
    missing_hours = set((df_target.loc[missing_mask, "hour"] + 1).tolist())

    candidates = {h: mae for h, mae in metrics["hourly_mae"].items() if mae is not None and h not in missing_hours}
    if not candidates:
        candidates = {h: mae for h, mae in metrics["hourly_mae"].items() if mae is not None}
    if not candidates:
        return None, None

    best_hour = min(candidates, key=candidates.get)
    return best_hour, candidates[best_hour]


def fit_final_model(df_hist, feature_cols):
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(df_hist[feature_cols], df_hist["lmp"])
    return model


def attach_reference_price(df, reference_df, target_date, forecast_csv_name, feature_name="dam_price"):
    """Same-hour reference price as a feature (e.g. DAM price, for the RTM/Spread models):
    the actual published price for history, and the matching predict_*.py script's own
    prediction for target_date, since the actual price doesn't exist there yet.
    Returns (df, used_forecast) -- used_forecast is False if that script hasn't been run."""
    df = df.merge(reference_df.rename(columns={"lmp": feature_name}), on="interval_start_local", how="left")

    forecast_path = DATA_DIR / forecast_csv_name
    used_forecast = forecast_path.exists()
    if used_forecast:
        forecast_df = pd.read_csv(forecast_path)
        hour_to_pred = dict(zip(forecast_df["hour"], forecast_df["predicted_lmp"]))
        target_mask = df["interval_start_local"].dt.date == target_date
        df.loc[target_mask, feature_name] = (df.loc[target_mask, "hour"] + 1).map(hour_to_pred)
    return df, used_forecast


def find_similar_day(df, df_hist, target_date):
    """Nearest historical day to target_date on the full set of forecasted variables we
    have for tomorrow (load forecast, wind forecast, weather forecast) plus weekend-ness.
    Returns (analog_date, distance, comparison) or None if too little data to compare.
    `comparison` holds each feature's target vs. analog-day value, for display."""
    agg = {col: (col, "mean") for col in ANALOG_FEATURE_COLS}
    daily = df.groupby(df["interval_start_local"].dt.date).agg(
        is_weekend=("is_weekend", "max"),
        n_hours=("interval_start_local", "count"),
        **agg,
    )
    if target_date not in daily.index or daily.loc[target_date, ANALOG_FEATURE_COLS].isna().any():
        return None

    complete_dates = set(df_hist["interval_start_local"].dt.date)
    daily_hist = daily[daily.index.isin(complete_dates) & (daily["n_hours"] == 24)].dropna(subset=ANALOG_FEATURE_COLS)
    daily_hist = daily_hist.drop(index=target_date, errors="ignore")
    if daily_hist.empty:
        return None

    mu = daily_hist[ANALOG_FEATURE_COLS].mean()
    sigma = daily_hist[ANALOG_FEATURE_COLS].std().replace(0, 1)
    z_hist = (daily_hist[ANALOG_FEATURE_COLS] - mu) / sigma
    z_target = (daily.loc[target_date, ANALOG_FEATURE_COLS] - mu) / sigma

    # Weekday/weekend demand shapes differ enough that a weekend analog for a
    # weekday target (or vice versa) should lose even if load/wind/weather happen to match.
    weekend_penalty = (daily_hist["is_weekend"] != daily.loc[target_date, "is_weekend"]).astype(float) * 3.0
    dist = np.sqrt(((z_hist - z_target) ** 2).sum(axis=1)) + weekend_penalty
    analog_date = dist.idxmin()

    comparison = {
        col: {"target": float(daily.loc[target_date, col]), "analog": float(daily_hist.loc[analog_date, col])}
        for col in ANALOG_FEATURE_COLS
    }
    return analog_date, float(dist.loc[analog_date]), comparison
