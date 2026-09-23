"""Tomorrow's hourly spread for one zone, in two parts, both on DART = DA - RT (the Spread
tab's convention: positive when day-ahead clears above real-time).

1. The point forecast, as before: run_forecast() on the DART series (data/spread_forecast.*),
   with the two most similar historical days, for the shape of the day.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from forecast_common import (
    DATA_DIR, SUPPLY_COLS, ZONE, add_lag_features, attach_reference_price, build_grid, dam_target_prediction,
    determine_target_date, forecast_file, information_cutoff, load_forecast_inputs, load_price_series,
    missing_inputs, parse_run_args, run_forecast, usable_feature_cols,
)

BIG_T = 40  # $/MWh: |DART| beyond this is a big hour (~10% of hours each side)
BACKTEST_DAYS = 60  # longer than the price models' 21: ~10% events need the hours
CONF_EDGES = [0.0, 0.35, 0.5, 0.65, 0.8, 1.01]  # P(DART > 0) bins scored for hit rate and edge
TIER_MIN_EDGE = {"high": 5.0, "medium": 1.0}  # bin $/h edge needed for each tier
TIER_MIN_HIT_HIGH = 60.0  # and "high" also needs this hit rate (%)
MIN_BIN_HOURS = 24  # fewer backtest hours than this and the bin is unrated (low)
MIN_T_STAT = 2.0  # and the edge must be >= this many standard errors from zero, or it's noise
WATCH_LIFT = 2.0  # fallback flag rule: P >= WATCH_LIFT x base rate, when the holdout can't set a threshold
WATCH_MIN_PRECISION = 0.25  # the flag threshold is the lowest P whose holdout precision clears this...
WATCH_MIN_HOURS = 6         # ...over at least this many flagged hours; maximises recall at that precision
MODEL_PARAMS = dict(max_depth=4, learning_rate=0.05, max_iter=300)

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
]

LOW_RT = 5.0  # $/MWh: RT collapsed to the floor -- the SBG regime (2026-09-21 HE4-5 was $0.5 / $0.0)

# Each target is a rule on the hourly frame: 'lmp' is DART, 'rt' the real-time price.
TARGETS = {
    "pos": lambda f: f["lmp"] > 0,
    "big_pos": lambda f: f["lmp"] > BIG_T,
    "big_neg": lambda f: f["lmp"] < -BIG_T,
    # The SBG watch: P(RT <= LOW_RT). Rare (1.7% of hours overall, 5-8% in spring) and very
    # asymmetric for a virtual load, so it gets its own classifier and flag; the supply-side
    # features (net_load in particular) are what make it learnable.
    "low_rt": lambda f: f["rt"] <= LOW_RT,
}
WATCHES = {"big_pos": "big_pos_watch", "big_neg": "big_neg_watch", "low_rt": "sbg_watch"}
CALL_POS, CALL_NEG = "DART > 0", "DART < 0"


def compute_spread_series(dam, rtm):
    """DART = DA - RT, the Spread tab's sign."""
    merged = dam.merge(rtm, on="interval_start_local", suffixes=("_dam", "_rtm"))
    merged["lmp"] = merged["lmp_dam"] - merged["lmp_rtm"]
    return merged[["interval_start_local", "lmp"]]


def fit_probas(train, X, cols):
    """One booster per target, all fit on train and scored on X. Returns {name: probas}."""
    out = {}
    for name, rule in TARGETS.items():
        y = rule(train).astype(int)
        model = HistGradientBoostingClassifier(random_state=0, **MODEL_PARAMS)
        model.fit(train[cols], y)
        out[name] = model.predict_proba(X[cols])[:, 1]
    return out


def watch_threshold(probs, actual, base):
    """The probability above which a big hour is flagged. Chosen on a holdout: the lowest
    threshold whose precision is >= WATCH_MIN_PRECISION, and at least WATCH_LIFT x that
    holdout's base rate, over >= WATCH_MIN_HOURS hours (i.e. the most recall that precision
    allows). Without the lift condition a holdout whose base rate itself clears 25% -- big
    positive DART was that common in parts of the summer -- sets the threshold at the bottom
    and flags every hour. P >= WATCH_LIFT x base rate is the fallback when no threshold
    qualifies. backtest() scores the choice on data it wasn't made on; scorecard.py has the
    archive's number."""
    order = np.argsort(-probs)
    hits = np.cumsum(actual[order])
    n = np.arange(1, len(probs) + 1)
    precision = hits / n
    need = max(WATCH_MIN_PRECISION, WATCH_LIFT * float(np.mean(actual)))
    ok = np.where((precision >= need) & (n >= WATCH_MIN_HOURS))[0]
    if len(ok):
        return float(probs[order][ok[-1]])
    return WATCH_LIFT * base


def bin_of(p):
    return int(np.searchsorted(CONF_EDGES, p, side="right") - 1)


def tier_of(hit_rate, edge, n, se=0.0):
    if hit_rate is None or n < MIN_BIN_HOURS or edge < MIN_T_STAT * se:
        return "low"
    if edge >= TIER_MIN_EDGE["high"] and hit_rate >= TIER_MIN_HIT_HIGH:
        return "high"
    return "medium" if edge >= TIER_MIN_EDGE["medium"] else "low"


def score_bins(p_pos, dart):
    """Per P(DART > 0) bin: the side that paid (sign of the mean DART), how often it was right,
    the edge and the tier. The call is whichever side paid; too few hours -> the likelier side."""
    bins = []
    for i in range(len(CONF_EDGES) - 1):
        m = np.array([bin_of(v) == i for v in p_pos], dtype=bool)
        n = int(m.sum())
        e_dart = float(dart[m].mean()) if n else 0.0
        se = float(dart[m].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        call_pos = e_dart > 0 if n >= MIN_BIN_HOURS else CONF_EDGES[i] >= 0.5
        hits = (dart[m] > 0) == call_pos
        hr = float(hits.mean() * 100) if n else None
        edge = abs(e_dart)
        bins.append({"lo": CONF_EDGES[i], "hi": min(CONF_EDGES[i + 1], 1.0), "n": n,
                     "call": CALL_POS if call_pos else CALL_NEG, "e_dart": round(e_dart, 2), "se": round(se, 2),
                     "hit_rate": None if hr is None else round(hr, 1),
                     "pnl": round(float(edge * n), 2), "edge": round(edge, 2),
                     "tier": tier_of(hr, edge, n, se)})
    return bins


def backtest(df_hist, cols):
    """Trailing holdout scored the way the signal is used: direction calls, their hit rate and
    1 MW P&L overall and per tier, Brier skill vs. the base rate, and the precision of the
    big-hour watch flags vs. their base rates.

    The bins' calls/tiers and the watch thresholds are *fit* on a holdout, so scoring them on
    that same holdout flatters them (until 2026-09-22 the page's hit rates and P&L were exactly
    that). The holdout is therefore split in two: both are fit on the older half and scored on
    the newer half -- the numbers reported, over eval_days. What the live signal uses is then
    refit on the whole holdout.

    The probabilities are not calibrated. An isotonic map fit on the trailing window was tried
    (P(DART > 0) runs overconfident: ~0.84 given, ~0.72 observed) and made the walk-forward
    Brier score worse, 0.2384 -> 0.2425 over 67 days: the window's calibration doesn't carry
    to the next day across regime changes. The calls don't need it -- each bin is judged by
    the DART it actually paid. Returns (metrics, base rates)."""
    cutoff = df_hist["interval_start_local"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    train = df_hist[df_hist["interval_start_local"] < cutoff]
    test = df_hist[df_hist["interval_start_local"] >= cutoff]
    if train.empty or test.empty:
        return None
    usable = usable_feature_cols(train, cols)
    p_raw = fit_probas(train, test, usable)
    dart = test["lmp"].values
    base = {k: float(rule(train).mean()) for k, rule in TARGETS.items()}

    # Fit on the older half, score on the newer one.
    fit_m = (test["interval_start_local"] < cutoff + pd.Timedelta(days=BACKTEST_DAYS / 2)).values
    ev_m = ~fit_m
    fit_frame, ev_frame = test[fit_m], test[ev_m]
    p_fit = {k: v[fit_m] for k, v in p_raw.items()}
    p_ev = {k: v[ev_m] for k, v in p_raw.items()}
    bins_half = score_bins(p_fit["pos"], dart[fit_m])
    dart_ev = dart[ev_m]
    bin_idx = np.array([bin_of(v) for v in p_ev["pos"]], dtype=int)
    call_pos = np.array([bins_half[i]["call"] == CALL_POS for i in bin_idx], dtype=bool)
    hit = call_pos == (dart_ev > 0)
    pnl = np.where(call_pos, dart_ev, -dart_ev)  # 1 MW in the called direction: gen when DART > 0, load when < 0
    high = np.array([bins_half[i]["tier"] == "high" for i in bin_idx], dtype=bool)
    rated = np.array([bins_half[i]["tier"] != "low" for i in bin_idx], dtype=bool)

    def brier_skill(name, probs):
        y = TARGETS[name](ev_frame).astype(int)
        return round(float(1 - brier_score_loss(y, probs) / brier_score_loss(y, np.full(len(ev_frame), base[name]))), 3)

    # What the live signal uses: refit on the whole holdout.
    p_all = p_raw
    bins = score_bins(p_all["pos"], dart)

    def watch(name):
        actual_fit = TARGETS[name](fit_frame).values
        actual_ev = TARGETS[name](ev_frame).values
        thr_half = watch_threshold(p_fit[name], actual_fit, base[name])
        flagged = p_ev[name] >= thr_half
        return {"threshold": round(float(watch_threshold(p_all[name], TARGETS[name](test).values, base[name])), 4),
                "n_flagged": int(flagged.sum()),
                "precision": round(float(actual_ev[flagged].mean() * 100), 1) if flagged.any() else None,
                "recall": round(float((actual_ev & flagged).sum() / actual_ev.sum() * 100), 1) if actual_ev.sum() else None,
                "base_rate": round(float(actual_ev.mean() * 100), 1)}

    metrics = {
        "days": BACKTEST_DAYS, "eval_days": BACKTEST_DAYS // 2, "n_hours": int(ev_m.sum()), "bins_hours": int(len(test)),
        "base_rate_pos": round(base["pos"] * 100, 1),
        "hit_rate_all": round(float(hit.mean() * 100), 1), "pnl_all": round(float(pnl.sum()), 2),
        "n_rated": int(rated.sum()),
        "hit_rate_rated": round(float(hit[rated].mean() * 100), 1) if rated.any() else None,
        "pnl_rated": round(float(pnl[rated].sum()), 2),
        "n_high": int(high.sum()),
        "hit_rate_high": round(float(hit[high].mean() * 100), 1) if high.any() else None,
        "pnl_high": round(float(pnl[high].sum()), 2),
        "optimal_pnl": round(float(np.abs(dart_ev).sum()), 2),
        "always_gen_pnl": round(float(dart_ev.sum()), 2),  # virtual gen every hour (DART > 0 pays)
        "always_load_pnl": round(float(-dart_ev.sum()), 2),
        "brier_skill": {k: brier_skill(k, p_ev[k]) for k in TARGETS},
        "bins": bins,
        "watch": {name: watch(name) for name in WATCHES},
    }
    return metrics, base


SIGNAL_COLS = ["hour", "p_pos", "p_big_pos", "p_big_neg", "p_low_rt", "call", "tier", "confidence", "edge",
               "big_pos_watch", "big_neg_watch", "sbg_watch"]


def archive(out, meta, out_dir=DATA_DIR, zone=ZONE):
    """One row set per (target_date, vintage), like forecast_common.archive_forecast."""
    path = Path(out_dir) / forecast_file("spread", "signal_history", zone)
    rows = out[SIGNAL_COLS].copy()
    rows.insert(0, "backfilled", meta.get("backfilled", False))
    rows.insert(0, "generated_at", meta["generated_at"])
    rows.insert(0, "vintage", meta["vintage"])
    rows.insert(0, "target_date", meta["target_date"])
    if path.exists():
        old = pd.read_csv(path)
        if "vintage" not in old.columns:
            old["vintage"] = "pre_dam"
        keep = ~((old["target_date"] == meta["target_date"]) & (old["vintage"] == meta["vintage"]))
        rows = pd.concat([old[keep], rows])
    rows.sort_values(["target_date", "vintage", "hour"]).to_csv(path, index=False)


def run_signal(target_date=None, write_latest=True, vintage="pre_dam", out_dir=DATA_DIR,
               dam_forecast_dir=DATA_DIR, zone=ZONE):
    """Signal for target_date (default: the live target for this vintage, anchored on the DAM
    like the price models). A past target_date is walk-forward: history is cut at that run's
    information cutoff and the DAM feature comes from the archived vintage in dam_forecast_dir,
    so a reconstruction reproduces what would have been shown."""
    dam = load_price_series("ieso_dam_prices.csv", zone)
    rtm = load_price_series("ieso_rtm_prices.csv", zone)
    dart = compute_spread_series(dam, rtm)
    load_fc, wind_fc, weather = load_forecast_inputs()
    backfill = target_date is not None
    tz = dart["interval_start_local"].dt.tz
    if backfill:
        dart = dart[dart["interval_start_local"] < information_cutoff(target_date, vintage, tz)]
        if vintage == "pre_dam":
            dam = dam[dam["interval_start_local"].dt.date < target_date]
        else:
            dam = dam[dam["interval_start_local"].dt.date <= target_date]
    else:
        target_date = determine_target_date(dam, vintage)

    df = build_grid(dart, load_fc, wind_fc, weather, target_date, tz)
    df = df.merge(rtm.rename(columns={"lmp": "rt"}), on="interval_start_local", how="left")
    target_pred = dam_target_prediction(target_date, vintage, dam_forecast_dir, zone)
    df, used_dam_forecast = attach_reference_price(df, dam, target_date, target_pred)
    df = add_lag_features(df, prefix="spread", vintage=vintage)
    df_hist = df[df["lmp"].notna()].copy()
    df_target = df[df["interval_start_local"].dt.date == target_date].copy()
    print(f"Signal target: {target_date} | training rows: {len(df_hist)} through {df_hist['interval_start_local'].max()}")

    cols_all = FEATURE_COLS
    metrics, base = backtest(df_hist, cols_all) or (None, None)
    if metrics and not backfill:
        print(f"Backtest (fit on {BACKTEST_DAYS - metrics['eval_days']}d, scored on the last {metrics['eval_days']}d, "
              f"{metrics['n_hours']}h): direction hit rate {metrics['hit_rate_all']}% "
              f"(base P(DART > 0) {metrics['base_rate_pos']}%), 1 MW P&L ${metrics['pnl_all']:.0f} of ${metrics['optimal_pnl']:.0f}; "
              f"high tier: {metrics['n_high']}h at {metrics['hit_rate_high']}%, ${metrics['pnl_high']:.0f}; "
              f"always gen ${metrics['always_gen_pnl']:.0f}, always load ${metrics['always_load_pnl']:.0f}")
        print(f"Brier skill: {metrics['brier_skill']} | watch: {metrics['watch']}")
        for b in metrics["bins"]:
            print(f"  P(DART > 0) {b['lo']:.2f}-{b['hi']:.2f}: n={b['n']:3d} E[DART]=${b['e_dart']}/h (±{b['se']}) -> {b['call']} hit={b['hit_rate']} -> {b['tier']}")

    usable = usable_feature_cols(df_hist, cols_all)
    p = fit_probas(df_hist, df_target, usable)
    missing, missing_cols = missing_inputs(df_target, cols_all) if not backfill else (0, {})

    bins = metrics["bins"] if metrics else []
    out = pd.DataFrame({"hour": (df_target["hour"] + 1).values})
    out["p_pos"] = np.round(p["pos"], 3)
    out["p_big_pos"] = np.round(p["big_pos"], 3)
    out["p_big_neg"] = np.round(p["big_neg"], 3)
    out["p_low_rt"] = np.round(p["low_rt"], 3)
    out["call"] = [bins[bin_of(v)]["call"] if bins else (CALL_POS if v >= 0.5 else CALL_NEG) for v in out["p_pos"]]
    out["confidence"] = [bins[bin_of(v)]["hit_rate"] if bins else None for v in out["p_pos"]]
    out["edge"] = [bins[bin_of(v)]["edge"] if bins else None for v in out["p_pos"]]
    out["tier"] = [bins[bin_of(v)]["tier"] if bins else "low" for v in out["p_pos"]]
    for name, col in WATCHES.items():
        thr = (metrics["watch"][name]["threshold"] if metrics else WATCH_LIFT * base[name]) if base else np.inf
        out[col] = out[f"p_{name}"] >= thr

    high = out[out["tier"] == "high"].sort_values("edge", ascending=False)
    meta = {
        "zone": zone,
        "target_date": str(target_date),
        "vintage": vintage,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "backfilled": backfill,
        "big_threshold": BIG_T, "watch_lift": WATCH_LIFT,
        "base_rates": {k: round(v * 100, 1) for k, v in base.items()} if base else None,
        "backtest": metrics,
        "top_hours": [{"hour": int(h), "call": c, "confidence": conf}
                      for h, c, conf in zip(high["hour"], high["call"], high["confidence"])],
        "missing_input_hours": missing,
        "missing_input_cols": missing_cols,
        "used_dam_forecast_feature": used_dam_forecast,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_latest:
        out_path = out_dir / forecast_file("spread", "signal", zone)
        out.to_csv(out_path, index=False)
        print(f"Saved {len(out)} rows to {out_path}")
        print("High-conviction hours:", ", ".join(f"HE{h} {c}" for h, c in zip(high["hour"], high["call"])) or "none")
        with open(out_dir / forecast_file("spread", "signal_meta", zone), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    archive(out, meta, out_dir=out_dir, zone=zone)
    return out, meta


def main(vintage="pre_dam", target_date=None, out_dir=DATA_DIR, dam_forecast_dir=DATA_DIR, backtest_enabled=True,
         zone=ZONE):
    dam = load_price_series("ieso_dam_prices.csv", zone)
    rtm = load_price_series("ieso_rtm_prices.csv", zone)
    run_forecast("spread", compute_spread_series(dam, rtm), FEATURE_COLS, dam, attach_dam_feature=True,
                 vintage=vintage, target_date=target_date, out_dir=out_dir, dam_forecast_dir=dam_forecast_dir,
                 backtest_enabled=backtest_enabled, zone=zone)
    return run_signal(target_date, write_latest=target_date is None, vintage=vintage, out_dir=out_dir,
                      dam_forecast_dir=dam_forecast_dir, zone=zone)


if __name__ == "__main__":
    args = parse_run_args("Forecast tomorrow's DART spread and the trading signal.")
    main(args.vintage, args.target_date, args.out_dir, zone=args.zone)
