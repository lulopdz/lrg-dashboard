"""Tomorrow's hourly spread for one zone, in two parts, both on DART = DA - RT (the Spread
tab's convention: positive when day-ahead clears above real-time).

1. The point forecast, as before: run_forecast() on the DART series (data/spread_forecast.*),
   with the two most similar historical days, for the shape of the day.
2. The SIGNAL: three classifiers on the same features, which is what a trader acts on:
     P(DART > 0)          direction
     P(DART > +BIG_T)     DA well above RT (a virtual gen pays)
     P(DART < -BIG_T)     RT well above DA (a virtual load pays)
   DART > 0 ~66% of hours in OTTAWA, so every probability is read against its base rate,
   and confidence is not the raw probability: each hour's P(DART > 0) falls in a bin, and
   the bin's hit rate and 1 MW P&L per hour over the trailing BACKTEST_DAYS are what we
   report. Tiers rank by $/h edge first, because the tails are one-sided (RT spikes, i.e.
   big negative DART, dwarf the positive side): a bin can be right 59% of the time and
   lose money, or right 47% and make it. "high" = paid >= $5/h AND right >= 60% of the
   time; "medium" = paid >= $1/h; anything else is "low", including bins too small to judge.

Writes data/spread_signal.csv, data/spread_signal_meta.json, and one vintage per target
day to data/spread_signal_history.csv. `--backfill` reconstructs that archive walk-forward.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from forecast_common import (
    DATA_DIR, ZONE, add_lag_features, attach_reference_price, build_grid, determine_target_date,
    load_forecast_inputs, load_price_series, run_forecast, usable_feature_cols,
)

BIG_T = 40  # $/MWh: |DART| beyond this is a big hour (~10% of hours each side)
BACKTEST_DAYS = 60  # longer than the price models' 21: ~10% events need the hours
CONF_EDGES = [0.0, 0.35, 0.5, 0.65, 0.8, 1.01]  # P(DART > 0) bins scored for hit rate and edge
TIER_MIN_EDGE = {"high": 5.0, "medium": 1.0}  # bin $/h edge needed for each tier
TIER_MIN_HIT_HIGH = 60.0  # and "high" also needs this hit rate (%)
MIN_BIN_HOURS = 24  # fewer backtest hours than this and the bin is unrated (low)
WATCH_LIFT = 2.0  # a big hour is flagged when its P >= WATCH_LIFT x base rate
MODEL_PARAMS = dict(max_depth=4, learning_rate=0.05, max_iter=300)

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "ontario", "ontario_northeast", "ontario_northwest", "ontario_southwest", "ontario_southeast",
    "wind_forecast",
    "temperature_2m", "relative_humidity_2m", "precipitation", "snowfall", "wind_speed_10m", "shortwave_radiation",
    "dam_price",
    "spread_lag_1d", "spread_lag_7d", "spread_roll_7d", "spread_roll_28d",
]

TARGETS = {
    "pos": lambda d: d > 0,
    "big_pos": lambda d: d > BIG_T,
    "big_neg": lambda d: d < -BIG_T,
}
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
    """Trailing holdout scored the way the signal is used: direction calls (P(DART > 0) >= 0.5),
    their hit rate and 1 MW P&L overall and per bin, Brier skill vs. the base rate, and the
    precision of the big-hour watch flags vs. their base rates."""
    cutoff = df_hist["interval_start_local"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    train = df_hist[df_hist["interval_start_local"] < cutoff]
    test = df_hist[df_hist["interval_start_local"] >= cutoff]
    if train.empty or test.empty:
        return None
    usable = usable_feature_cols(train, cols)
    p = fit_probas(train, test, usable)
    dart = test["lmp"].values
    base = {k: float(rule(train["lmp"]).mean()) for k, rule in TARGETS.items()}

    call_pos = p["pos"] >= 0.5
    hit = call_pos == (dart > 0)
    pnl = np.where(call_pos, dart, -dart)  # 1 MW in the called direction: gen when DART > 0, load when < 0
    bins = []
    for i in range(len(CONF_EDGES) - 1):
        m = np.array([bin_of(v) == i for v in p["pos"]])
        n = int(m.sum())
        hr = float(hit[m].mean() * 100) if n else None
        edge = float(pnl[m].mean()) if n else 0.0
        bins.append({"lo": CONF_EDGES[i], "hi": min(CONF_EDGES[i + 1], 1.0), "n": n,
                     "hit_rate": None if hr is None else round(hr, 1),
                     "pnl": round(float(pnl[m].sum()), 2) if n else 0.0, "edge": round(edge, 2),
                     "tier": tier_of(hr, edge, n)})
    tier_by_bin = [b["tier"] for b in bins]
    high = np.array([tier_by_bin[bin_of(v)] == "high" for v in p["pos"]])

    def brier_skill(name):
        y = TARGETS[name](test["lmp"]).astype(int)
        return round(float(1 - brier_score_loss(y, p[name]) / brier_score_loss(y, np.full(len(test), base[name]))), 3)

    def watch(name):
        flagged = p[name] >= WATCH_LIFT * base[name]
        actual = TARGETS[name](test["lmp"]).values
        return {"n_flagged": int(flagged.sum()),
                "precision": round(float(actual[flagged].mean() * 100), 1) if flagged.any() else None,
                "base_rate": round(float(actual.mean() * 100), 1)}

    return {
        "days": BACKTEST_DAYS, "n_hours": int(len(test)),
        "base_rate_pos": round(base["pos"] * 100, 1),
        "hit_rate_all": round(float(hit.mean() * 100), 1), "pnl_all": round(float(pnl.sum()), 2),
        "n_high": int(high.sum()),
        "hit_rate_high": round(float(hit[high].mean() * 100), 1) if high.any() else None,
        "pnl_high": round(float(pnl[high].sum()), 2),
        "optimal_pnl": round(float(np.abs(dart).sum()), 2),
        "always_gen_pnl": round(float(dart.sum()), 2),  # virtual gen every hour (DART > 0 pays)
        "always_load_pnl": round(float(-dart.sum()), 2),
        "brier_skill": {k: brier_skill(k) for k in TARGETS},
        "bins": bins,
        "watch": {"big_pos": watch("big_pos"), "big_neg": watch("big_neg")},
    }, base


def archive(out, meta):
    path = DATA_DIR / "spread_signal_history.csv"
    rows = out[["hour", "p_pos", "p_big_pos", "p_big_neg", "call", "tier", "big_pos_watch", "big_neg_watch"]].copy()
    rows.insert(0, "backfilled", meta.get("backfilled", False))
    rows.insert(0, "generated_at", meta["generated_at"])
    rows.insert(0, "target_date", meta["target_date"])
    if path.exists():
        old = pd.read_csv(path)
        rows = pd.concat([old[old["target_date"] != meta["target_date"]], rows])
    rows.sort_values(["target_date", "hour"]).to_csv(path, index=False)


def archived_dam_vintage(target_date):
    """The archived DAM forecast for a past target day (feature + timestamp for backfill)."""
    hist = DATA_DIR / "dam_forecast_history.csv"
    if not hist.exists():
        return {}, None
    h = pd.read_csv(hist)
    h = h[h["target_date"] == str(target_date)]
    return dict(zip(h["hour"], h["predicted_lmp"])), (h["generated_at"].iloc[0] if len(h) else None)


def run_signal(target_date=None, write_latest=True):
    """Signal for target_date (default: tomorrow, anchored on the DAM like the price models).
    A past target_date is walk-forward: history is cut at that day and the DAM feature comes
    from the archived vintage, so backfill_signal() reproduces what would have been shown."""
    dam = load_price_series("ieso_dam_prices.csv")
    rtm = load_price_series("ieso_rtm_prices.csv")
    dart = compute_spread_series(dam, rtm)
    load_fc, wind_fc, weather = load_forecast_inputs()
    backfill = target_date is not None
    target_date = target_date or determine_target_date(dam)
    tz = dart["interval_start_local"].dt.tz
    if backfill:
        dart = dart[dart["interval_start_local"].dt.date < target_date]

    df = build_grid(dart, load_fc, wind_fc, weather, target_date, tz)
    dam_generated_at = None
    if backfill:
        dam_pred, dam_generated_at = archived_dam_vintage(target_date)
        df = df[df["interval_start_local"].dt.date <= target_date]
        df = df.merge(dam.rename(columns={"lmp": "dam_price"}), on="interval_start_local", how="left")
        mask = df["interval_start_local"].dt.date == target_date
        df.loc[mask, "dam_price"] = (df.loc[mask, "hour"] + 1).map(dam_pred)
        used_dam_forecast = bool(dam_pred)
    else:
        df, used_dam_forecast = attach_reference_price(df, dam, target_date, "dam_forecast.csv")
    df = add_lag_features(df, prefix="spread")
    df_hist = df[df["lmp"].notna()].copy()
    df_target = df[df["interval_start_local"].dt.date == target_date].copy()
    print(f"Signal target: {target_date} | training rows: {len(df_hist)} through {df_hist['interval_start_local'].max()}")

    metrics, base = backtest(df_hist, FEATURE_COLS) or (None, None)
    if metrics and not backfill:
        print(f"Backtest ({BACKTEST_DAYS}d, {metrics['n_hours']}h): direction hit rate {metrics['hit_rate_all']}% "
              f"(base P(DART > 0) {metrics['base_rate_pos']}%), 1 MW P&L ${metrics['pnl_all']:.0f} of ${metrics['optimal_pnl']:.0f}; "
              f"high tier: {metrics['n_high']}h at {metrics['hit_rate_high']}%, ${metrics['pnl_high']:.0f}; "
              f"always gen ${metrics['always_gen_pnl']:.0f}, always load ${metrics['always_load_pnl']:.0f}")
        print(f"Brier skill: {metrics['brier_skill']} | watch: {metrics['watch']}")
        for b in metrics["bins"]:
            print(f"  P(DART > 0) {b['lo']:.2f}-{b['hi']:.2f}: n={b['n']:3d} hit={b['hit_rate']} edge=${b['edge']}/h -> {b['tier']}")

    usable = usable_feature_cols(df_hist, FEATURE_COLS)
    p = fit_probas(df_hist, df_target, usable)
    missing = int(df_target[FEATURE_COLS].isna().any(axis=1).sum())
    if missing and not backfill:
        print(f"Warning: {missing} of 24 target hours have missing inputs.")

    bins = metrics["bins"] if metrics else []
    out = pd.DataFrame({"hour": (df_target["hour"] + 1).values})
    out["p_pos"] = np.round(p["pos"], 3)
    out["p_big_pos"] = np.round(p["big_pos"], 3)
    out["p_big_neg"] = np.round(p["big_neg"], 3)
    out["call"] = np.where(out["p_pos"] >= 0.5, CALL_POS, CALL_NEG)
    out["confidence"] = [bins[bin_of(v)]["hit_rate"] if bins else None for v in out["p_pos"]]
    out["edge"] = [bins[bin_of(v)]["edge"] if bins else None for v in out["p_pos"]]
    out["tier"] = [bins[bin_of(v)]["tier"] if bins else "low" for v in out["p_pos"]]
    out["big_pos_watch"] = (out["p_big_pos"] >= WATCH_LIFT * base["big_pos"]) if base else False
    out["big_neg_watch"] = (out["p_big_neg"] >= WATCH_LIFT * base["big_neg"]) if base else False

    high = out[out["tier"] == "high"].sort_values("edge", ascending=False)
    meta = {
        "zone": ZONE,
        "target_date": str(target_date),
        # A reconstruction is stamped with its DAM vintage's time: when this signal would have run.
        "generated_at": (dam_generated_at or pd.Timestamp.now(tz="UTC").isoformat()) if backfill
                        else pd.Timestamp.now(tz="UTC").isoformat(),
        "backfilled": backfill,
        "big_threshold": BIG_T, "watch_lift": WATCH_LIFT,
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
        with open(DATA_DIR / "spread_signal_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    archive(out, meta)
    return out, meta


def backfill_signal():
    """One-off: walk-forward signal for every day in dam_forecast_history.csv."""
    hist = pd.read_csv(DATA_DIR / "dam_forecast_history.csv")
    for d in sorted(hist["target_date"].unique()):
        run_signal(pd.Timestamp(d).date(), write_latest=False)


def main():
    dam = load_price_series("ieso_dam_prices.csv")
    rtm = load_price_series("ieso_rtm_prices.csv")
    run_forecast("spread", compute_spread_series(dam, rtm), FEATURE_COLS, dam, attach_dam_feature=True)
    run_signal()


if __name__ == "__main__":
    import sys
    if "--backfill" in sys.argv:
        backfill_signal()
    else:
        main()
