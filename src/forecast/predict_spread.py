"""Tomorrow's hourly spread SIGNAL for one zone: direction and opportunity probabilities
instead of a point forecast of RT - DA.

A point forecast of the spread compounds the DAM and RTM errors and answers a question a
trader doesn't ask. What matters is (1) which way it goes and (2) in which hours the call is
worth acting on. So this fits three classifiers on the same features as predict_dam/rtm.py:
  P(up)    = P(RT > DA)              direction
  P(spike) = P(RT - DA > +SPIKE_T)   RT well above DA (pays a virtual load)
  P(dip)   = P(RT - DA < -DIP_T)     RT well below DA (pays a virtual gen)
RT > DA only ~34% of hours in OTTAWA, so every probability is read against its base rate,
and confidence is not the raw probability: each hour's P(up) falls in a bin, and the bin's
hit rate and 1 MW P&L per hour over the trailing BACKTEST_DAYS are what we report. Tiers
need both: the spread's tails are fat and one-sided (RT spikes dwarf RT dips), so a bin can
be right 59% of the time and lose money, or right 47% and make it. Tiers therefore rank
by $/h edge first: "high" = calls like this one paid >= $5/h AND were right >= 60% of the
time over the window; "medium" = paid >= $1/h; anything else is "low", including bins
with too few hours to judge.

Writes data/spread_signal.csv, data/spread_signal_meta.json, and appends one vintage per
target day to data/spread_signal_history.csv.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from forecast_common import (
    DATA_DIR, ZONE, add_lag_features, attach_reference_price, build_grid, determine_target_date,
    load_forecast_inputs, load_price_series, usable_feature_cols,
)

SPIKE_T = 40  # $/MWh: RT - DA above this is a spike; below -DIP_T a dip (~10% of hours each)
DIP_T = 40
BACKTEST_DAYS = 60  # longer than the price models' 21: ~10% events need the hours
CONF_EDGES = [0.0, 0.2, 0.35, 0.5, 0.65, 1.01]  # P(up) bins scored for hit rate
TIER_MIN_EDGE = {"high": 5.0, "medium": 1.0}  # bin $/h edge needed for each tier
TIER_MIN_HIT_HIGH = 60.0  # and "high" also needs this hit rate (%)
MIN_BIN_HOURS = 24  # fewer backtest hours than this and the bin is unrated (low)
WATCH_LIFT = 2.0  # spike/dip flagged when P >= WATCH_LIFT x its base rate
MODEL_PARAMS = dict(max_depth=4, learning_rate=0.05, max_iter=300)

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "dam_price",
    "delta_lag_1d", "delta_lag_7d", "delta_roll_7d", "delta_roll_28d",
]

TARGETS = {
    "up": lambda d: d > 0,
    "spike": lambda d: d > SPIKE_T,
    "dip": lambda d: d < -DIP_T,
}


def compute_delta_series(dam, rtm):
    """RT - DA, positive when real-time clears above day-ahead (the opposite sign of the
    Spread tab's DAM - RTM, on purpose: 'RT > DA' reads as up)."""
    merged = dam.merge(rtm, on="interval_start_local", suffixes=("_dam", "_rtm"))
    merged["lmp"] = merged["lmp_rtm"] - merged["lmp_dam"]
    return merged[["interval_start_local", "lmp"]]


def fit_probas(train, X, cols):
    """One booster per target, all fit on train and scored on X. Returns {name: probas}."""
    out = {}
    for name, rule in TARGETS.items():
        y = rule(train["lmp"]).astype(int)
        model = HistGradientBoostingClassifier(random_state=0, **MODEL_PARAMS)
        model.fit(train[cols], y)
        out[name] = model.predict_proba(X[cols])[:, 1]
    return out


def bin_of(p):
    return int(np.searchsorted(CONF_EDGES, p, side="right") - 1)


def tier_of(hit_rate, edge, n):
    if hit_rate is None or n < MIN_BIN_HOURS:
        return "low"
    if edge >= TIER_MIN_EDGE["high"] and hit_rate >= TIER_MIN_HIT_HIGH:
        return "high"
    return "medium" if edge >= TIER_MIN_EDGE["medium"] else "low"


def backtest(df_hist, cols):
    """Trailing holdout scored the way the signal is used: direction calls (P(up) >= 0.5),
    their hit rate and 1 MW P&L overall and per P(up) bin, Brier skill vs. the base rate, and
    precision of the spike/dip watch flags vs. their base rates."""
    cutoff = df_hist["interval_start_local"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    train = df_hist[df_hist["interval_start_local"] < cutoff]
    test = df_hist[df_hist["interval_start_local"] >= cutoff]
    if train.empty or test.empty:
        return None
    usable = usable_feature_cols(train, cols)
    p = fit_probas(train, test, usable)
    delta = test["lmp"].values
    base = {k: float(rule(train["lmp"]).mean()) for k, rule in TARGETS.items()}

    call_up = p["up"] >= 0.5
    hit = call_up == (delta > 0)
    pnl = np.where(call_up, delta, -delta)
    bins = []
    for i in range(len(CONF_EDGES) - 1):
        m = np.array([bin_of(v) == i for v in p["up"]])
        n = int(m.sum())
        hr = float(hit[m].mean() * 100) if n else None
        edge = float(pnl[m].mean()) if n else 0.0  # $/MWh per hour traded in the called direction
        bins.append({"lo": CONF_EDGES[i], "hi": min(CONF_EDGES[i + 1], 1.0), "n": n,
                     "hit_rate": None if hr is None else round(hr, 1),
                     "pnl": round(float(pnl[m].sum()), 2) if n else 0.0, "edge": round(edge, 2),
                     "tier": tier_of(hr, edge, n)})
    tier_by_bin = [b["tier"] for b in bins]
    high = np.array([tier_by_bin[bin_of(v)] == "high" for v in p["up"]])

    def brier_skill(name):
        bs = brier_score_loss(TARGETS[name](test["lmp"]).astype(int), p[name])
        ref = brier_score_loss(TARGETS[name](test["lmp"]).astype(int), np.full(len(test), base[name]))
        return round(float(1 - bs / ref), 3)

    def watch(name):
        flagged = p[name] >= WATCH_LIFT * base[name]
        actual = TARGETS[name](test["lmp"]).values
        return {"n_flagged": int(flagged.sum()),
                "precision": round(float(actual[flagged].mean() * 100), 1) if flagged.any() else None,
                "base_rate": round(float(actual.mean() * 100), 1)}

    return {
        "days": BACKTEST_DAYS, "n_hours": int(len(test)),
        "base_rate_up": round(base["up"] * 100, 1),
        "hit_rate_all": round(float(hit.mean() * 100), 1), "pnl_all": round(float(pnl.sum()), 2),
        "n_high": int(high.sum()),
        "hit_rate_high": round(float(hit[high].mean() * 100), 1) if high.any() else None,
        "pnl_high": round(float(pnl[high].sum()), 2),
        "optimal_pnl": round(float(np.abs(delta).sum()), 2),
        "always_load_pnl": round(float(delta.sum()), 2),  # virtual load every hour (RT > DA pays)
        "always_gen_pnl": round(float(-delta.sum()), 2),  # virtual gen every hour
        "brier_skill": {k: brier_skill(k) for k in TARGETS},
        "bins": bins,
        "watch": {"spike": watch("spike"), "dip": watch("dip")},
    }, base


def expected_dam(target_date=None):
    """DAM point forecast and per-hour error band for the chart's price row: the live
    predict_dam.py output, or the archived vintage for a past target_date. ({} if absent.)"""
    if target_date is None:
        csv, meta = DATA_DIR / "dam_forecast.csv", DATA_DIR / "dam_forecast_meta.json"
        if not (csv.exists() and meta.exists()):
            return {}, {}, None
        df = pd.read_csv(csv)
        mae = (json.load(open(meta, encoding="utf-8")).get("backtest") or {}).get("hourly_mae") or {}
        return dict(zip(df["hour"], df["predicted_lmp"])), {int(h): v for h, v in mae.items()}, None
    hist = DATA_DIR / "dam_forecast_history.csv"
    if not hist.exists():
        return {}, {}, None
    h = pd.read_csv(hist)
    h = h[h["target_date"] == str(target_date)]
    return dict(zip(h["hour"], h["predicted_lmp"])), dict(zip(h["hour"], h["band_err"])),         (h["generated_at"].iloc[0] if len(h) else None)


def archive(out, meta):
    path = DATA_DIR / "spread_signal_history.csv"
    rows = out[["hour", "p_up", "p_spike", "p_dip", "call", "tier", "spike_watch", "dip_watch"]].copy()
    rows.insert(0, "backfilled", meta.get("backfilled", False))
    rows.insert(0, "generated_at", meta["generated_at"])
    rows.insert(0, "target_date", meta["target_date"])
    if path.exists():
        old = pd.read_csv(path)
        rows = pd.concat([old[old["target_date"] != meta["target_date"]], rows])
    rows.sort_values(["target_date", "hour"]).to_csv(path, index=False)


def run(target_date=None, write_latest=True):
    """Signal for target_date (default: tomorrow, anchored on the DAM like the price models).
    A past target_date is walk-forward: history is cut at that day and the DAM feature comes
    from the archived vintage, so backfill_signal() reproduces what would have been shown."""
    dam = load_price_series("ieso_dam_prices.csv")
    rtm = load_price_series("ieso_rtm_prices.csv")
    delta = compute_delta_series(dam, rtm)
    load_fc, wind_fc, weather = load_forecast_inputs()
    backfill = target_date is not None
    target_date = target_date or determine_target_date(dam)
    tz = delta["interval_start_local"].dt.tz
    if backfill:
        delta = delta[delta["interval_start_local"].dt.date < target_date]

    df = build_grid(delta, load_fc, wind_fc, weather, target_date, tz)
    dam_pred, dam_mae, dam_generated_at = expected_dam(target_date if backfill else None)
    if backfill:
        df = df[df["interval_start_local"].dt.date <= target_date]
        df = df.merge(dam.rename(columns={"lmp": "dam_price"}), on="interval_start_local", how="left")
        mask = df["interval_start_local"].dt.date == target_date
        df.loc[mask, "dam_price"] = (df.loc[mask, "hour"] + 1).map(dam_pred)
        used_dam_forecast = bool(dam_pred)
    else:
        df, used_dam_forecast = attach_reference_price(df, dam, target_date, "dam_forecast.csv")
    df = add_lag_features(df, prefix="delta")
    df_hist = df[df["lmp"].notna()].copy()
    df_target = df[df["interval_start_local"].dt.date == target_date].copy()
    print(f"Target date: {target_date} | training rows: {len(df_hist)} through {df_hist['interval_start_local'].max()}")

    metrics, base = backtest(df_hist, FEATURE_COLS) or (None, None)
    if metrics and not backfill:
        print(f"Backtest ({BACKTEST_DAYS}d, {metrics['n_hours']}h): direction hit rate {metrics['hit_rate_all']}% "
              f"(base P(up) {metrics['base_rate_up']}%), 1 MW P&L ${metrics['pnl_all']:.0f} of ${metrics['optimal_pnl']:.0f}; "
              f"high tier: {metrics['n_high']}h at {metrics['hit_rate_high']}%, ${metrics['pnl_high']:.0f}; "
              f"always load ${metrics['always_load_pnl']:.0f}, always gen ${metrics['always_gen_pnl']:.0f}")
        print(f"Brier skill: {metrics['brier_skill']} | watch: {metrics['watch']}")
        for b in metrics["bins"]:
            print(f"  P(up) {b['lo']:.2f}-{b['hi']:.2f}: n={b['n']:3d} hit={b['hit_rate']} edge=${b['edge']}/h -> {b['tier']}")

    usable = usable_feature_cols(df_hist, FEATURE_COLS)
    p = fit_probas(df_hist, df_target, usable)
    missing = int(df_target[FEATURE_COLS].isna().any(axis=1).sum())
    if missing and not backfill:
        print(f"Warning: {missing} of 24 target hours have missing inputs.")

    bins = metrics["bins"] if metrics else []
    out = pd.DataFrame({"hour": (df_target["hour"] + 1).values})
    out["p_up"] = np.round(p["up"], 3)
    out["p_spike"] = np.round(p["spike"], 3)
    out["p_dip"] = np.round(p["dip"], 3)
    out["call"] = np.where(out["p_up"] >= 0.5, "RT > DA", "RT < DA")
    out["confidence"] = [bins[bin_of(v)]["hit_rate"] if bins else None for v in out["p_up"]]
    out["edge"] = [bins[bin_of(v)]["edge"] if bins else None for v in out["p_up"]]
    out["tier"] = [bins[bin_of(v)]["tier"] if bins else "low" for v in out["p_up"]]
    out["spike_watch"] = (out["p_spike"] >= WATCH_LIFT * base["spike"]) if base else False
    out["dip_watch"] = (out["p_dip"] >= WATCH_LIFT * base["dip"]) if base else False
    out["expected_dam"] = out["hour"].map(dam_pred)
    out["dam_band"] = out["hour"].map(dam_mae)

    high = out[out["tier"] == "high"].sort_values("edge", ascending=False)
    meta = {
        "zone": ZONE,
        "target_date": str(target_date),
        # A reconstruction is stamped with its DAM vintage's time: when this signal would have run.
        "generated_at": (dam_generated_at or pd.Timestamp.now(tz="UTC").isoformat()) if backfill
                        else pd.Timestamp.now(tz="UTC").isoformat(),
        "backfilled": backfill,
        "spike_threshold": SPIKE_T, "dip_threshold": DIP_T, "watch_lift": WATCH_LIFT,
        "base_rates": {k: round(v * 100, 1) for k, v in base.items()} if base else None,
        "backtest": metrics,
        "top_hours": [{"hour": int(h), "call": c, "confidence": conf}
                      for h, c, conf in zip(high["hour"], high["call"], high["confidence"])],
        "missing_input_hours": missing,
        "used_dam_forecast_feature": used_dam_forecast,
    }
    if write_latest:
        out_path = DATA_DIR / "spread_signal.csv"
        out.to_csv(out_path, index=False)
        print(f"Saved {len(out)} rows to {out_path}")
        print("High-conviction hours:", ", ".join(f"HE{h} {c}" for h, c in zip(high["hour"], high["call"])) or "none")
        meta_path = DATA_DIR / "spread_signal_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {meta_path}")
    archive(out, meta)
    return out, meta


def backfill_signal():
    """One-off: walk-forward signal for every day in dam_forecast_history.csv (the days a
    DAM forecast was archived), so the Day picker has history from day one."""
    hist = pd.read_csv(DATA_DIR / "dam_forecast_history.csv")
    for d in sorted(hist["target_date"].unique()):
        run(pd.Timestamp(d).date(), write_latest=False)


if __name__ == "__main__":
    import sys
    if "--backfill" in sys.argv:
        backfill_signal()
    else:
        run()
