"""Similar-day analog search and comparison formatting helpers."""
import numpy as np
import pandas as pd

from config import (
    _ANALOG_SPEC,
    ANALOG_FEATURE_COLS,
    ANALOG_FEATURE_LABELS,
    ANALOG_FEATURE_WEIGHTS,
)


def find_similar_day(df, df_hist, target_date, n=2):
    """The n historical days closest to target_date (closest first) on the full set of
    forecasted variables we have for tomorrow (load forecast, wind forecast, weather
    forecast) plus weekend-ness, weighted by ANALOG_FEATURE_WEIGHTS so load/wind/
    temperature count more than the rest. Returns a list of up to n dicts
    {date, distance, comparison} (empty if too little data to compare). `comparison`
    holds each feature's target vs. that day's value, for display."""
    agg = {key: (src, func) for key, src, func, _label, _unit, _w in _ANALOG_SPEC}
    daily = df.groupby(df["interval_start_local"].dt.date).agg(
        is_weekend=("is_weekend", "max"),
        n_hours=("interval_start_local", "count"),
        **agg,
    )
    if target_date not in daily.index or daily.loc[target_date, ANALOG_FEATURE_COLS].isna().any():
        return []

    complete_dates = set(df_hist["interval_start_local"].dt.date)
    daily_hist = daily[daily.index.isin(complete_dates) & (daily["n_hours"] == 24)].dropna(subset=ANALOG_FEATURE_COLS)
    daily_hist = daily_hist.drop(index=target_date, errors="ignore")
    if daily_hist.empty:
        return []

    mu = daily_hist[ANALOG_FEATURE_COLS].mean()
    sigma = daily_hist[ANALOG_FEATURE_COLS].std().replace(0, 1)
    z_hist = (daily_hist[ANALOG_FEATURE_COLS] - mu) / sigma
    z_target = (daily.loc[target_date, ANALOG_FEATURE_COLS] - mu) / sigma
    weights = pd.Series(ANALOG_FEATURE_WEIGHTS)[ANALOG_FEATURE_COLS]

    # Weekday/weekend demand shapes differ enough that a weekend analog for a
    # weekday target (or vice versa) should lose even if load/wind/weather happen to match.
    weekend_penalty = (daily_hist["is_weekend"] != daily.loc[target_date, "is_weekend"]).astype(float) * 3.0
    dist = np.sqrt((weights * (z_hist - z_target) ** 2).sum(axis=1)) + weekend_penalty

    analogs = []
    for date, distance in dist.nsmallest(n).items():
        comparison = {
            col: {"target": float(daily.loc[target_date, col]), "analog": float(daily_hist.loc[date, col])}
            for col in ANALOG_FEATURE_COLS
        }
        analogs.append({"date": date, "distance": float(distance), "comparison": comparison})
    return analogs


def analog_comparison_display(analogs, rank):
    """Format one ranked analog's feature comparison list for forecast metadata JSON."""
    if len(analogs) <= rank:
        return []
    comparison = analogs[rank]["comparison"]
    return [
        {
            "label": label,
            "unit": unit,
            "weight": ANALOG_FEATURE_WEIGHTS[col],
            "target": comparison[col]["target"],
            "analog": comparison[col]["analog"],
        }
        for col, (label, unit) in ANALOG_FEATURE_LABELS.items()
    ]


def analog_price_curves(analogs, df_hist):
    """Return [{hour (1-24): lmp}] curves for each analog day in `analogs`."""
    curves = []
    for a in analogs:
        rows = df_hist[df_hist["interval_start_local"].dt.date == a["date"]]
        curves.append(dict(zip((rows["hour"] + 1).tolist(), rows["lmp"].tolist())))
    return curves
