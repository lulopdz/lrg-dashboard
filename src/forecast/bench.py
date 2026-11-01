"""Fast bench for DART point-forecast ideas: minutes per idea instead of walkforward.py's ~80.

    python src/forecast/bench.py --variants rt_clip:200,direct_l2         # against the shipped model
    python src/forecast/bench.py --variants rt_clip:200 --zones OTTAWA --refit 7
    python src/forecast/bench.py --variants direct_l1 --save /tmp/b.pkl   # keep the forecasts

What it does differently from walkforward.py, and why that's fine for screening: one hourly grid
per zone, and the models are refit every --refit days (default 14) on an expanding window cut a
day before each block, instead of every day. Within each block it fits, like a pre_dam run:
the DAM model (its forecast is the serve-time dam_price), the shipped RT model, and the
P(DART > 0) classifier the picks need. Every variant's DART forecast is then scored as 5 picks
a day (picks_5) and as the 5 largest |forecast| (top5_fc), per zone and per segment (days with
the 9:00 inputs vs the latest ones), and compared with the baseline per day.

Calibration (2026-11-01, four zones, 2025-07-19..2026-10-31): the shipped model ("current")
made $14,243 on picks_5 here on the 9:00-input days where walkforward.py made $15,894, and
$96,074 vs $96,217 on the rest -- the same totals, but the refit cadence alone moves a single
zone by up to +-30% (NIAGARA, 9:00 inputs: $3,451 vs $4,803). So: screen ideas here, keep the
ones that win clearly and in most zones, and confirm the survivor with walkforward.py --all-days
before shipping (README, "Evaluar un cambio").

Adding an idea: write a function (ctx) -> DART forecast array and register it in VARIANTS (or
in PARAMETRIC for a "name:value" family). ctx carries the train/test frames of both grids
(rt_tr/rt_te: RT series; sp_tr/sp_te: DART series), their feature lists and dam_fc."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_dam  # noqa: E402
import predict_rtm  # noqa: E402
import predict_spread  # noqa: E402
import scorecard  # noqa: E402
import strategies  # noqa: E402
from forecast_common import (DAYAHEAD_PATH, DEFAULT_MODEL_PARAMS, ZONES, add_lag_features, build_grid,  # noqa: E402
                             load_forecast_inputs, load_price_series, training_target, usable_feature_cols,
                             zone_feature_cols)
from picks import top_picks  # noqa: E402

FIRST_DAY = pd.Timestamp("2025-07-19")  # MIN_TRAIN_DAYS after the first DAM day, like walkforward --all-days


def fit_predict(train, test, cols, y, params=None, **kw):
    cols = usable_feature_cols(train, cols)
    m = HistGradientBoostingRegressor(random_state=0, **{**(DEFAULT_MODEL_PARAMS if params is None else params), **kw})
    m.fit(train[cols], y)
    return m.predict(test[cols])


# --- variants: (ctx) -> DART forecast for the test rows ------------------------------------

def rt_based(ctx, y, **kw):
    """DAM forecast - RT forecast, the shipped structure, with the RT model fit on y."""
    return ctx["dam_fc"] - fit_predict(ctx["rt_tr"], ctx["rt_te"], ctx["rt_cols"], y, **kw)


def direct(ctx, y, **kw):
    """A booster on DART itself."""
    return fit_predict(ctx["sp_tr"], ctx["sp_te"], ctx["sp_cols"], y, **kw)


VARIANTS = {
    # What production ships (predict_rtm.TARGET_CLIP). The default baseline.
    "current": lambda c: rt_based(c, training_target(c["rt_tr"], predict_rtm.TARGET_CLIP)),
    "rt_noclip": lambda c: rt_based(c, c["rt_tr"]["lmp"].values),
    "rt_l1": lambda c: rt_based(c, c["rt_tr"]["lmp"].values, loss="absolute_error"),
    "direct_l2": lambda c: direct(c, c["sp_tr"]["lmp"].values),
    "direct_l1": lambda c: direct(c, c["sp_tr"]["lmp"].values, loss="absolute_error"),
}
# "name:value" families, e.g. rt_clip:200 or direct_clip:100.
PARAMETRIC = {
    "rt_clip": lambda c, v: rt_based(c, np.clip(c["rt_tr"]["lmp"].values, -v, v)),
    "direct_clip": lambda c, v: direct(c, np.clip(c["sp_tr"]["lmp"].values, -v, v)),
}


def variant_fn(name):
    if name in VARIANTS:
        return VARIANTS[name]
    family, _, value = name.partition(":")
    if family in PARAMETRIC and value:
        return lambda c: PARAMETRIC[family](c, float(value))
    raise SystemExit(f"Unknown variant {name!r}: one of {sorted(VARIANTS)} or {sorted(PARAMETRIC)}:<value>")


# --- the run ---------------------------------------------------------------------------------

def grids(zone):
    """The DAM, RT and DART hourly grids for one zone, with lags and the real DAM as dam_price."""
    dam = load_price_series("ieso_dam_prices.csv", zone)
    rtm = load_price_series("ieso_rtm_prices.csv", zone)
    load_fc, wind_fc, weather = load_forecast_inputs()
    tz = rtm["interval_start_local"].dt.tz
    last = rtm["interval_start_local"].max().date()
    merged = dam.merge(rtm, on="interval_start_local", suffixes=("_dam", "_rt"))
    dart = merged.assign(lmp=merged["lmp_dam"] - merged["lmp_rt"])[["interval_start_local", "lmp"]]
    out = {}
    for prefix, series in (("dam", dam), ("rtm", rtm), ("spread", dart)):
        g = build_grid(series, load_fc, wind_fc, weather, last, tz)
        if prefix != "dam":
            g = g.merge(dam.rename(columns={"lmp": "dam_price"}), on="interval_start_local", how="left")
        out[prefix] = add_lag_features(g, prefix=prefix, known_through_yesterday=(prefix == "dam"), vintage="pre_dam")
    return out


def run_zone(zone, names, refit):
    g = grids(zone)
    dm, rt, sp = g["dam"], g["rtm"], g["spread"]
    cols = {"dam": zone_feature_cols(predict_dam.FEATURE_COLS, zone), "rt": zone_feature_cols(predict_rtm.FEATURE_COLS, zone),
            "sp": zone_feature_cols(predict_spread.FEATURE_COLS, zone)}
    # Each grid gets its own day index: they need not start at the same hour.
    dday, rday, sday = (x["interval_start_local"].dt.tz_localize(None).dt.normalize() for x in (dm, rt, sp))
    last_rt = rday[rt["lmp"].notna()].max()
    test_days = sorted(d for d in rday.unique() if FIRST_DAY <= d <= last_rt)
    fns = {n: variant_fn(n) for n in names}
    rows = []
    for i in range(0, len(test_days), refit):
        block = test_days[i:i + refit]
        cut = block[0] - pd.Timedelta(days=1)
        dm_tr, dm_te = dm[(dday < cut) & dm["lmp"].notna()], dm[dday.isin(block)]
        dam_fc = pd.Series(fit_predict(dm_tr, dm_te, cols["dam"], dm_tr["lmp"].values, params=predict_dam.MODEL_PARAMS),
                           index=dm_te["interval_start_local"])
        # Serve time: dam_price is the forecast, matched by timestamp.
        rt_te = rt[rday.isin(block)].pipe(lambda x: x.assign(dam_price=x["interval_start_local"].map(dam_fc)))
        sp_te = sp[sday.isin(block)].pipe(lambda x: x.assign(dam_price=x["interval_start_local"].map(dam_fc)))
        assert (rt_te["interval_start_local"].values == sp_te["interval_start_local"].values).all(), "RT and DART grids out of step"
        sp_tr = sp[(sday < cut) & sp["lmp"].notna()]
        ctx = {"rt_tr": rt[(rday < cut) & rt["lmp"].notna()], "rt_te": rt_te, "sp_tr": sp_tr, "sp_te": sp_te,
               "rt_cols": cols["rt"], "sp_cols": cols["sp"], "dam_fc": rt_te["dam_price"].values}
        clf_cols = usable_feature_cols(sp_tr, cols["sp"])
        clf = HistGradientBoostingClassifier(random_state=0, **predict_spread.MODEL_PARAMS).fit(sp_tr[clf_cols], (sp_tr["lmp"] > 0).astype(int))
        ts = rt_te["interval_start_local"].dt.tz_localize(None)  # market time (.values would be UTC)
        res = pd.DataFrame({"zone": zone, "target_date": ts.dt.strftime("%Y-%m-%d").values, "hour": (ts.dt.hour + 1).values,
                            "dart": sp_te["lmp"].values, "dam_ref": ctx["dam_fc"], "p_pos": clf.predict_proba(sp_te[clf_cols])[:, 1]})
        for n, fn in fns.items():
            res[n] = fn(ctx)
        rows.append(res)
        print(f"  {zone}: block {i // refit + 1}/{-(-len(test_days) // refit)} from {block[0].date()}", flush=True)
    return pd.concat(rows, ignore_index=True)


# --- scoring ---------------------------------------------------------------------------------

def strategy_frame(d, name):
    """d's rows as strategies.py wants them, with `name` as the forecast and its picks."""
    f = d.dropna(subset=["dart"]).rename(columns={name: "fc"})[["target_date", "hour", "dart", "fc", "p_pos", "dam_ref", "segment"]]
    f["pick"] = 0
    for _, idx in f.groupby("target_date").indices.items():
        f.iloc[idx, f.columns.get_loc("pick")] = top_picks(f["fc"].values[idx], f["p_pos"].values[idx])
    return f


def report(d, names, baseline):
    covered = set(pd.read_csv(DAYAHEAD_PATH, usecols=["target_date"])["target_date"]) if DAYAHEAD_PATH.exists() else set()
    d = d.assign(segment=np.where(d["target_date"].isin(covered), "9am_inputs", "latest_inputs"))
    zones = list(dict.fromkeys(d["zone"]))
    print("\n=== P&L at 1 MW (picks_5 | top5_fc), by segment ===")
    for seg in ("9am_inputs", "latest_inputs"):
        print(f"--- {seg}")
        for n in names:
            cells = []
            for z in zones:
                f = strategy_frame(d[(d["zone"] == z) & (d["segment"] == seg)], n)
                s = strategies.score(f) if len(f) else None
                cells.append(f"{z} {s['picks_5']['pnl']:>7.0f} | {s['top5_fc']['pnl']:>7.0f}" if s else f"{z} -")
            print(f"  {n:16s} " + "   ".join(cells))
    for cand in [n for n in names if n != baseline]:
        for seg in ("9am_inputs", "latest_inputs", None):
            per_zone = {}
            for z in zones:
                dz = d[d["zone"] == z] if seg is None else d[(d["zone"] == z) & (d["segment"] == seg)]
                if len(dz):
                    per_zone[z] = {"pre_dam": strategies.compare(strategy_frame(dz, baseline), strategy_frame(dz, cand))}
            label = "all days" if seg is None else seg
            lines = scorecard.verdict(per_zone).splitlines()
            print(f"\n{cand} vs {baseline}, {label}:")
            print("\n".join(l for l in lines[1:] if " picks_5 " in l or " top5_fc " in l))


def main():
    ap = argparse.ArgumentParser(description="Screen DART forecast variants against the shipped model.")
    ap.add_argument("--variants", required=True, help=f"comma list: {', '.join(VARIANTS)}, or {', '.join(PARAMETRIC)}:<value>")
    ap.add_argument("--baseline", default="current")
    ap.add_argument("--zones", default="all", help=f"comma list or 'all' ({', '.join(ZONES)})")
    ap.add_argument("--refit", type=int, default=14, help="days between refits (1 = like production, ~6x slower)")
    ap.add_argument("--save", default=None, help="pickle the per-hour forecasts here")
    args = ap.parse_args()
    names = list(dict.fromkeys([args.baseline, *[v.strip() for v in args.variants.split(",") if v.strip()]]))
    for n in names:
        variant_fn(n)  # fail fast on a typo, before minutes of fitting
    zones = ZONES if args.zones == "all" else tuple(args.zones.split(","))
    t0 = time.time()
    d = pd.concat([run_zone(z, names, args.refit) for z in zones], ignore_index=True)
    if args.save:
        d.to_pickle(args.save)
    report(d, names, args.baseline)
    print(f"\n{len(zones)} zones, {d['target_date'].nunique()} days, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
