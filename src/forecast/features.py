"""Calendar, solar ramp, causal lag, and reference-price feature engineering."""
import holidays
import numpy as np
import pandas as pd

from config import OTTAWA_LAT, OTTAWA_LON, RT_HOURS_KNOWN
from io_archive import load_supply_inputs, overlay_dayahead


def sunset_hour(dates):
    """Ottawa's sunset in market time (EST, fractional hours) for each date -- the NOAA
    approximation, good to a few minutes, which is plenty for an hourly feature."""
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    b = 2 * np.pi * (doy - 81) / 364
    eqtime = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)  # minutes
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    lat = np.radians(OTTAWA_LAT)
    cos_h = (np.sin(np.radians(-0.833)) - np.sin(lat) * np.sin(decl)) / (np.cos(lat) * np.cos(decl))
    half_day = np.degrees(np.arccos(np.clip(cos_h, -1, 1))) / 15
    noon = 12 + (-75 - OTTAWA_LON) / 15 - eqtime / 60  # EST's meridian is 75 W
    return noon + half_day


def build_grid(target_df, load_fc, wind_fc, weather, target_date, tz, supply=None):
    """One continuous hourly grid spanning history through target_date, with the target
    series ('lmp'), forecast inputs, supply-side inputs and calendar features merged in.
    'lmp' is NaN for target_date -- that's the row(s) to predict."""
    start = target_df["interval_start_local"].min()
    end = pd.Timestamp(target_date, tz=tz) + pd.Timedelta(hours=23)
    idx = pd.date_range(start, end, freq="h", tz=tz)

    df = pd.DataFrame({"interval_start_local": idx})
    df = df.merge(target_df, on="interval_start_local", how="left")
    df = df.merge(load_fc, on="interval_start_local", how="left")
    df = df.merge(wind_fc, on="interval_start_local", how="left")
    df = df.merge(weather, on="interval_start_local", how="left")
    if supply is None:
        supply = load_supply_inputs()
    if supply is not None:
        df = df.merge(supply, on="interval_start_local", how="left")
    df = overlay_dayahead(df)

    df["hour"] = df["interval_start_local"].dt.hour
    df["dow"] = df["interval_start_local"].dt.dayofweek
    df["month"] = df["interval_start_local"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # Ontario statutory holidays trade like a Sunday (Labour Day 2026 cleared ~$15 below the
    # Monday before it); is_weekend alone can't tell the model that.
    dates = df["interval_start_local"].dt.date
    on_holidays = holidays.CA(subdiv="ON", years=range(dates.min().year, dates.max().year + 1))
    df["is_holiday"] = dates.map(lambda d: d in on_holidays).astype(int)
    for col, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    # The evening ramp (RAMP_COLS): the mid-hour against sunset, and the forecast load, net load
    # and solar one hour either side -- all forecasts, so the target day has them like history.
    # The grid ends at the target day's HE24, which has no next hour: its ramp out repeats its
    # ramp in (solar: 0, it's night), so the last hour isn't flagged as a missing input every day.
    day = df["interval_start_local"].dt.normalize()
    df["hours_from_sunset"] = df["hour"] + 0.5 - day.map(dict(zip(day.unique(), sunset_hour(day.unique().tz_localize(None)))))
    for src, name in (("ontario", "load"), ("net_load", "net_load")):
        if src in df.columns:
            df[f"{name}_ramp_in"] = df[src] - df[src].shift(1)
            df[f"{name}_ramp_out"] = (df[src].shift(-1) - df[src]).fillna(df[f"{name}_ramp_in"])
    if "solar_forecast" in df.columns:
        df["solar_ramp_out"] = (df["solar_forecast"].shift(-1) - df["solar_forecast"]).fillna(0)

    return df


def add_lag_features(df, prefix, known_through_yesterday=False, vintage="pre_dam"):
    """Causal lags on df['lmp'], shifted by whole days on the complete hourly grid so nothing
    at hour t ever sees data from t or later -- and, just as important, nothing the run
    doesn't have yet when it happens.

    known_through_yesterday=True (DAM: published a full day at once, so the whole of today
    is known at the 9:00 run) keeps the same-hour lag at 1 day. False (RT and the spread:
    today's hours are still being settled) shifts every lag one more day, and instead
    summarises today so far -- the mean and max of the RT_HOURS_KNOWN[vintage] - 1 hours
    that are complete by run time -- as {prefix}_today_early_mean/max, applied to every hour
    of the target day. Before this, rtm_lag_1d was NaN for ~16 of the 24 target hours at
    serve time and NaN in 0.2% of training rows: the booster answered those hours from a
    branch it had never trained."""
    lmp = df["lmp"]
    shift_days = 1 if known_through_yesterday else 2
    df[f"{prefix}_lag_{shift_days}d"] = lmp.shift(24 * shift_days)
    df[f"{prefix}_lag_7d"] = lmp.shift(24 * 7)
    df[f"{prefix}_roll_7d"] = lmp.shift(24 * shift_days).rolling(24 * 7, min_periods=24 * 3).mean()
    df[f"{prefix}_roll_28d"] = lmp.shift(24 * shift_days).rolling(24 * 28, min_periods=24 * 7).mean()
    if not known_through_yesterday:
        # One hour short of RT_HOURS_KNOWN on purpose: that constant is the summer case (the
        # 9:00 Ottawa run is 8:00 EST, so hour 6 closed only an hour earlier), and a late RT
        # publication or an early trigger must not leave the feature computed from a different
        # set of hours live than in training. In winter (9:00 EST) the margin is two hours.
        early_hours = RT_HOURS_KNOWN[vintage] - 1
        day = df["interval_start_local"].dt.normalize()
        early = df[df["hour"] < early_hours].groupby(day)["lmp"].agg(["mean", "max"])
        prev_day = day - pd.Timedelta(days=1)
        df[f"{prefix}_today_early_mean"] = prev_day.map(early["mean"])
        df[f"{prefix}_today_early_max"] = prev_day.map(early["max"])
    return df


def usable_feature_cols(df, feature_cols):
    """Drop any feature that's entirely missing in this slice. HistGradientBoostingRegressor
    handles partial missingness fine, but a column with zero non-null values can crash its
    histogram binning step on some scikit-learn versions -- e.g. a forecast source (like the
    Ontario-wide wind forecast) that only has a couple weeks of history won't have a single
    real value in an older training window, even though it's fully populated more recently."""
    return [c for c in feature_cols if df[c].notna().any()]


def missing_inputs(df_target, feature_cols):
    """(hours with any missing input, {column: missing hours}) for the target day. The
    booster still predicts through NaN, but along a branch trained on few rows -- e.g. the
    same-day RT summary when the 9:00 RT download failed -- so the meta records which
    inputs were missing and the page says so."""
    na = df_target[feature_cols].isna()
    n_hours = int(na.any(axis=1).sum())
    cols = {c: int(v) for c, v in na.sum().items() if v}
    if n_hours:
        print(f"Warning: {n_hours} of {len(df_target)} target hours have missing inputs "
              f"({', '.join(f'{c} x{v}' for c, v in cols.items())}) -- predictions for those hours are less reliable.")
    return n_hours, cols


def attach_reference_price(df, reference_df, target_date, target_pred=None, feature_name="dam_price"):
    """Same-hour reference price as a feature (e.g. DAM price, for the RTM/Spread models):
    the actual published price for history, and -- when the actual price doesn't exist yet
    for target_date -- the matching predict_*.py script's prediction, passed as target_pred
    ({hour 1-24: value}). None means "use whatever the reference series has" (post_dam runs,
    where tomorrow's DAM is real). Returns (df, used_forecast).

    History keeps the real DAM even though a pre_dam run is served a forecast one. Training on
    out-of-fold DAM forecasts instead (month by month, a DAM model fit on earlier data) was
    tried on 2026-09-22: walk-forward RTM MAE $22.72 -> $23.60 and the spread signal's rated
    1 MW P&L $2,093 -> $258 over 67 days, so it was dropped."""
    df = df.merge(reference_df.rename(columns={"lmp": feature_name}), on="interval_start_local", how="left")
    target_mask = df["interval_start_local"].dt.date == target_date
    if target_pred is not None:
        df.loc[target_mask, feature_name] = (df.loc[target_mask, "hour"] + 1).map(target_pred)
    return df, target_pred is not None
