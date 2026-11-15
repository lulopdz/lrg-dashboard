"""Tomorrow's hourly spread for one zone, in two parts, both on DART = DA - RT (the Spread
tab's convention: positive when day-ahead clears above real-time).

1. The point forecast (data/spread_forecast.*): DAM forecast - RT forecast, from the same run
   and vintage (post_dam: the real DAM - RT forecast), with the two most similar days' DART
   for the shape of the day. See derived_forecast().
2. The SIGNAL: three classifiers on the same features, which is what a trader acts on:
     P(DART > 0)          direction
     P(DART > +BIG_T)     DA well above RT (a virtual gen pays)
     P(DART < -BIG_T)     RT well above DA (a virtual load pays)
   DART > 0 ~66% of hours in OTTAWA (every hour of the day), so "the likelier side" is
   nearly always DART > 0 and is not the question. The call is the side that has PAID in
   hours like this one: each hour's P(DART > 0) falls in a bin, and over the trailing
   BACKTEST_DAYS the bin's mean DART decides the call (its sign) and the edge (its size,
   $/h for 1 MW). The tails are one-sided (RT spikes, i.e. big negative DART, dwarf the
   positive side), so a bin where DART > 0 happens 61% of the time can still pay to trade
   DART < 0. Tiers: "high" = edge >= $5/h AND the call right >= 60% of the time; "medium" =
   edge >= $1/h; anything else is "low" (no call), including bins too small to judge.

Writes data/spread_signal.csv, data/spread_signal_meta.json, and one row set per target day
and vintage to data/spread_signal_history.csv. Past days are reconstructed walk-forward by
walkforward.py (or `--target-date YYYY-MM-DD` here).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, RAMP_COLS, SUPPLY_COLS, ZONE, forecast_file, parse_run_args
from io_archive import archive_forecast, dam_target_prediction, determine_target_date, load_price_series
from spread_signal import (
    BACKTEST_DAYS,
    BIG_T,
    CALL_NEG,
    CALL_POS,
    CONF_EDGES,
    LOW_RT,
    MIN_BIN_HOURS,
    MIN_T_STAT,
    MODEL_PARAMS,
    SIGNAL_COLS,
    TARGETS,
    TIER_MIN_EDGE,
    TIER_MIN_HIT_HIGH,
    WATCH_LIFT,
    WATCH_MIN_HOURS,
    WATCH_MIN_PRECISION,
    WATCHES,
    archive,
    backtest,
    bin_of,
    compute_spread_series,
    fit_probas,
    run_signal,
    score_bins,
    tier_of,
    watch_threshold,
)

# Unlike the DAM/RTM models: no raw `hour` and no hub-height wind (wind_speed_100m_port_alma).
# Adding both was checked walk-forward over 67 days (2026-09-22): point MAE $22.46 -> $22.20
# (noise), the signal's rated 1 MW P&L $2,093 -> $1,012. Left out on purpose.
FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "is_holiday",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "dam_price",
    "spread_lag_2d", "spread_lag_7d", "spread_roll_7d", "spread_roll_28d", "spread_today_early_mean", "spread_today_early_max",
    *SUPPLY_COLS,  # nuclear/gas/hydro availability, net load, capacity margin (forecast_common)
    *RAMP_COLS,  # the evening ramp: hours from sunset, load/net-load/solar ramps (forecast_common)
]


def rtm_run(target_date, vintage, zone, dirs):
    """The RT forecast rows (hour, predicted_lmp, p10, p90, analog dates) archived for
    target_date and vintage, from the first of dirs that has them."""
    for d in dirs:
        path = Path(d) / forecast_file("rtm", "forecast_history", zone)
        if not path.exists():
            continue
        h = pd.read_csv(path)
        if "vintage" not in h.columns:
            h["vintage"] = "pre_dam"
        h = h[(h["target_date"] == str(target_date)) & (h["vintage"] == vintage)].sort_values("hour")
        if len(h):
            return h
    raise RuntimeError(f"No {vintage} RT forecast archived for {target_date} ({zone}) in {dirs}: run predict_rtm.py first")


def derived_forecast(vintage="pre_dam", target_date=None, out_dir=DATA_DIR, dam_forecast_dir=DATA_DIR, zone=ZONE):
    """DART point forecast = DAM forecast - RT forecast, instead of a third booster trained on
    DART itself. Scored on the archive (2026-07-10..09-23, 68 days, pre_dam) the booster got
    the sign right 55.8% of hours and following it made $5,693 at 1 MW; the difference of the
    two price forecasts got 62.1% and $8,369, at the same MAE ($22.84 vs $23.07). Walk-forward
    said the same (57.8% vs 56.1%, $4,844 vs $3,971). Switched 2026-09-23.

    pre_dam uses predict_dam.py's forecast for the DAM (the archived pre_dam one for
    reconstructions); post_dam uses the real DAM, so there DART's uncertainty is RT's alone.
    The band is the RT band carried over: P10 = DAM - RT's P90, P90 = DAM - RT's P10 (exact for
    post_dam; for pre_dam it leaves out the DAM forecast's own, much smaller, error). The
    similar days are the RT model's, with the DART that cleared on them. Writes and archives
    the same files as before (spread_forecast*.csv/json, spread_forecast_history*.csv)."""
    dam = load_price_series("ieso_dam_prices.csv", zone)
    rtm = load_price_series("ieso_rtm_prices.csv", zone)
    reconstruction = target_date is not None
    if not reconstruction:
        target_date = determine_target_date(dam, vintage)
    rt = rtm_run(target_date, vintage, zone, [out_dir, DATA_DIR])
    if vintage == "post_dam":
        day = dam[dam["interval_start_local"].dt.date == target_date]
        dam_fc = dict(zip(day["interval_start_local"].dt.hour + 1, day["lmp"]))
    else:
        dam_fc = dam_target_prediction(target_date, "pre_dam", dam_forecast_dir, zone) or {}
    if not dam_fc:
        raise RuntimeError(f"No DAM {'price' if vintage == 'post_dam' else 'forecast'} for {target_date} ({zone})")

    dart = compute_spread_series(dam, rtm)
    dart_by_day = {d: dict(zip(g["interval_start_local"].dt.hour + 1, g["lmp"]))
                   for d, g in dart.groupby(dart["interval_start_local"].dt.date.astype(str))}
    hours = rt["hour"].astype(int).values
    dam_h = np.array([dam_fc.get(h, np.nan) for h in hours], dtype=float)
    out = pd.DataFrame({"hour": hours, "predicted_lmp": np.round(dam_h - rt["predicted_lmp"].values, 2)})
    if {"p10", "p90"} <= set(rt.columns) and rt["p10"].notna().any():
        out["p10"] = np.round(dam_h - rt["p90"].values, 2)
        out["p90"] = np.round(dam_h - rt["p10"].values, 2)
    analog_dates = [rt[c].iloc[0] if c in rt.columns and pd.notna(rt[c].iloc[0]) else None for c in ("analog_date", "analog_date_2")]
    for col, d in zip(("analog_lmp", "analog_lmp_2"), analog_dates):
        out[col] = out["hour"].map(dart_by_day.get(str(d), {}) if d else {}).astype(float).round(2)

    meta = {
        "zone": zone,
        "target_date": str(target_date),
        "vintage": vintage,
        "reconstruction": reconstruction,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "DAM forecast - RT forecast" if vintage == "pre_dam" else "real DAM - RT forecast",
        "rtm_generated_at": rt["generated_at"].iloc[0],
        "missing_input_hours": int(np.isnan(dam_h).sum() + rt["predicted_lmp"].isna().sum()),
        "analog_date": analog_dates[0],
        "analog_date_2": analog_dates[1],
        "backtest": None,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / forecast_file("spread", "forecast", zone), index=False)
    with open(out_dir / forecast_file("spread", "forecast_meta", zone), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    archive_forecast("spread", out, meta, out_dir=out_dir, zone=zone)
    print(f"DART forecast {target_date} ({vintage}) = {meta['method']}: "
          f"{', '.join(f'HE{h} {v:+.1f}' for h, v in zip(out['hour'][:6], out['predicted_lmp'][:6]))}, ...")
    return out, meta


def main(vintage="pre_dam", target_date=None, out_dir=DATA_DIR, dam_forecast_dir=DATA_DIR, backtest_enabled=True,
         zone=ZONE):
    """The DART point forecast (derived_forecast), then the signal. Needs this run's RT forecast
    archived first (predict_rtm.py; run_forecasts.py and walkforward.py keep that order).
    backtest_enabled is kept for walkforward.py's signature; the point forecast has no
    backtest of its own and the signal's is part of how its calls are made."""
    forecast, _ = derived_forecast(vintage, target_date, out_dir, dam_forecast_dir, zone)
    return run_signal(target_date, write_latest=target_date is None, vintage=vintage, out_dir=out_dir,
                      dam_forecast_dir=dam_forecast_dir, zone=zone, forecast=forecast, feature_cols=FEATURE_COLS)


__all__ = [
    "BACKTEST_DAYS",
    "BIG_T",
    "CALL_NEG",
    "CALL_POS",
    "CONF_EDGES",
    "FEATURE_COLS",
    "LOW_RT",
    "MIN_BIN_HOURS",
    "MIN_T_STAT",
    "MODEL_PARAMS",
    "SIGNAL_COLS",
    "TARGETS",
    "TIER_MIN_EDGE",
    "TIER_MIN_HIT_HIGH",
    "WATCHES",
    "WATCH_LIFT",
    "WATCH_MIN_HOURS",
    "WATCH_MIN_PRECISION",
    "archive",
    "backtest",
    "bin_of",
    "compute_spread_series",
    "derived_forecast",
    "fit_probas",
    "main",
    "rtm_run",
    "run_signal",
    "score_bins",
    "tier_of",
    "watch_threshold",
]


if __name__ == "__main__":
    args = parse_run_args("Forecast tomorrow's DART spread and the trading signal.")
    main(args.vintage, args.target_date, args.out_dir, zone=args.zone)
