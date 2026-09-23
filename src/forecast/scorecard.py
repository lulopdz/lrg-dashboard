"""Out-of-sample scorecard for the archived forecasts: what each model actually predicted for
a day, scored against what then cleared. This is the number that decides whether a change
ships -- run_forecast()'s trailing 21-day backtest trains and tests on the same 'latest'
vintages of every input, so it is optimistic by construction, while the archive holds the
real 9:00 vintages the bot had (see forecast_common.archive_forecast).

    python src/forecast/scorecard.py                 # score data/*_history.csv, write data/forecast_scorecard.json
    python src/forecast/scorecard.py --dir A         # score histories in another folder (walkforward.py output)
    python src/forecast/scorecard.py --dir A --compare B   # side by side: A (baseline) vs B (candidate)

Every history row carries a `vintage` ("pre_dam": the 9:00 run, before tomorrow's DAM is
published -- what a virtual bid at the 10:00 deadline can use; "post_dam": the afternoon
re-run with the real DAM). Older archives without the column are scored as pre_dam."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_common import DATA_DIR, ZONE, forecast_file, load_price_series, zone_suffix

PREFIXES = ("dam", "rtm", "spread")
LOW_RT = 5.0     # $/MWh: the SBG tail (RT collapsed to ~0)
HIGH_RT = 150.0  # $/MWh: the spike tail
BIG_T = 40.0     # $/MWh: predict_spread's big-hour threshold, for scoring the watch flags


def _actuals(zone=ZONE):
    dam = load_price_series("ieso_dam_prices.csv", zone)
    rtm = load_price_series("ieso_rtm_prices.csv", zone)
    spread = dam.merge(rtm, on="interval_start_local", suffixes=("_dam", "_rtm"))
    spread["lmp"] = spread["lmp_dam"] - spread["lmp_rtm"]
    out = {}
    for name, df in (("dam", dam), ("rtm", rtm), ("spread", spread[["interval_start_local", "lmp"]])):
        df = df.copy()
        df["target_date"] = df["interval_start_local"].dt.strftime("%Y-%m-%d")
        df["hour"] = df["interval_start_local"].dt.hour + 1
        out[name] = df[["target_date", "hour", "lmp"]]
    return out


def _load_history(directory, prefix, zone=ZONE):
    path = Path(directory) / forecast_file(prefix, "forecast_history", zone)
    if not path.exists():
        return None
    h = pd.read_csv(path)
    if "vintage" not in h.columns:
        h["vintage"] = "pre_dam"
    return h


def score_prices(directory, actuals, zone=ZONE):
    """Per prefix and vintage: error stats, per-hour/month MAE, band coverage, analog MAE, and
    the baselines each model has to beat (RTM: 'RT = predicted DAM' and 'RT = real DAM')."""
    out = {}
    dam_hist = _load_history(directory, "dam", zone)
    for prefix in PREFIXES:
        h = _load_history(directory, prefix, zone)
        if h is None:
            continue
        h = h.merge(actuals[prefix], on=["target_date", "hour"], how="inner")
        if prefix == "rtm" and dam_hist is not None:
            dam_pred = dam_hist[dam_hist["vintage"] == "pre_dam"][["target_date", "hour", "predicted_lmp"]]
            h = h.merge(dam_pred.rename(columns={"predicted_lmp": "dam_pred"}), on=["target_date", "hour"], how="left")
            h = h.merge(actuals["dam"].rename(columns={"lmp": "dam_real"}), on=["target_date", "hour"], how="left")
        for vintage, g in h.groupby("vintage"):
            err = g["predicted_lmp"] - g["lmp"]
            rec = {
                "n_days": int(g["target_date"].nunique()), "n_hours": int(len(g)),
                "first_day": g["target_date"].min(), "last_day": g["target_date"].max(),
                "mae": round(float(err.abs().mean()), 2), "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
                "bias": round(float(err.mean()), 2), "actual_std": round(float(g["lmp"].std()), 2),
                "mae_by_hour": {int(k): round(float(v), 1) for k, v in err.abs().groupby(g["hour"]).mean().items()},
                "mae_by_month": {k: round(float(v), 1) for k, v in err.abs().groupby(g["target_date"].str[:7]).mean().items()},
                "analog_mae": (round(float((g["analog_lmp"] - g["lmp"]).abs().mean()), 2)
                               if "analog_lmp" in g and g["analog_lmp"].notna().any() else None),
            }
            if {"p10", "p90"} <= set(g.columns) and g["p10"].notna().any():
                inside = (g["lmp"] >= g["p10"]) & (g["lmp"] <= g["p90"])
                rec["band_p10_p90_coverage"] = round(float(inside.mean() * 100), 1)  # well calibrated ~ 80%
                rec["band_width_mean"] = round(float((g["p90"] - g["p10"]).mean()), 1)
            if prefix == "rtm":
                if "dam_pred" in g and g["dam_pred"].notna().any():
                    rec["baseline_rt_eq_dam_pred_mae"] = round(float((g["dam_pred"] - g["lmp"]).abs().mean()), 2)
                if "dam_real" in g:
                    rec["baseline_rt_eq_dam_real_mae"] = round(float((g["dam_real"] - g["lmp"]).abs().mean()), 2)
                rec["tail_low_actual_pct"] = round(float((g["lmp"] <= LOW_RT).mean() * 100), 2)
                rec["tail_low_predicted_pct"] = round(float((g["predicted_lmp"] <= LOW_RT).mean() * 100), 2)
                rec["tail_high_actual_pct"] = round(float((g["lmp"] > HIGH_RT).mean() * 100), 2)
                rec["tail_high_predicted_pct"] = round(float((g["predicted_lmp"] > HIGH_RT).mean() * 100), 2)
            if prefix == "spread":
                sign_ok = np.sign(g["predicted_lmp"]) == np.sign(g["lmp"])
                pnl = np.where(g["predicted_lmp"] > 0, g["lmp"], -g["lmp"])
                rec["sign_hit_rate"] = round(float(sign_ok.mean() * 100), 1)
                rec["base_rate_pos"] = round(float((g["lmp"] > 0).mean() * 100), 1)
                rec["pnl_follow_sign"] = round(float(pnl.sum()), 0)
                rec["pnl_always_gen"] = round(float(g["lmp"].sum()), 0)
                rec["pnl_optimal"] = round(float(g["lmp"].abs().sum()), 0)
                # The alternative to a third booster: the spread implied by the two price
                # forecasts, DAM_pred - RT_pred (same vintage). Scored here so the choice
                # needs no extra reconstruction.
                derived = _derived_spread(directory, vintage, zone)
                if derived is not None:
                    d = g.merge(derived, on=["target_date", "hour"], how="inner")
                    if len(d):
                        rec["derived_mae"] = round(float((d["derived"] - d["lmp"]).abs().mean()), 2)
                        rec["derived_sign_hit_rate"] = round(float((np.sign(d["derived"]) == np.sign(d["lmp"])).mean() * 100), 1)
                        rec["derived_pnl_follow_sign"] = round(float(np.where(d["derived"] > 0, d["lmp"], -d["lmp"]).sum()), 0)
            out.setdefault(prefix, {})[vintage] = rec
    return out


def _derived_spread(directory, vintage, zone=ZONE):
    dam, rtm = _load_history(directory, "dam", zone), _load_history(directory, "rtm", zone)
    if dam is None or rtm is None:
        return None
    dam = dam[dam["vintage"] == "pre_dam"][["target_date", "hour", "predicted_lmp"]]
    rtm = rtm[rtm["vintage"] == vintage][["target_date", "hour", "predicted_lmp"]]
    d = dam.merge(rtm, on=["target_date", "hour"], suffixes=("_dam", "_rtm"))
    d["derived"] = d["predicted_lmp_dam"] - d["predicted_lmp_rtm"]
    return d[["target_date", "hour", "derived"]]


def score_signal(directory, actuals, zone=ZONE):
    """The spread signal scored the way it is traded: hit rate and 1 MW P&L per tier, the
    watch flags' precision/recall against their base rates, and P(DART > 0) calibration.
    Returns {vintage: {"live" | "backfilled": record}}."""
    path = Path(directory) / forecast_file("spread", "signal_history", zone)
    if not path.exists():
        return None
    s = pd.read_csv(path)
    if "vintage" not in s.columns:
        s["vintage"] = "pre_dam"
    s = s.merge(actuals["spread"].rename(columns={"lmp": "dart"}), on=["target_date", "hour"], how="inner")
    s = s.merge(actuals["rtm"].rename(columns={"lmp": "rt"}), on=["target_date", "hour"], how="left")
    # Walk-forward reconstructions (backfilled) see the freshest version of every input, not the
    # 9:00 one, so they flatter the signal; scored apart from the days it was actually live.
    backfilled = s["backfilled"].astype(str).str.lower().eq("true") if "backfilled" in s else pd.Series(False, index=s.index)
    s["source"] = np.where(backfilled, "backfilled", "live")
    out = {}
    for (vintage, source), g in s.groupby(["vintage", "source"]):
        call_pos = g["call"].eq("DART > 0")
        hit = (call_pos == (g["dart"] > 0)).values
        pnl = np.where(call_pos, g["dart"], -g["dart"])
        rec = {"n_days": int(g["target_date"].nunique()), "n_hours": int(len(g)),
               "base_rate_pos": round(float((g["dart"] > 0).mean() * 100), 1),
               "pnl_optimal": round(float(g["dart"].abs().sum()), 0), "pnl_always_gen": round(float(g["dart"].sum()), 0),
               "tiers": {}}
        tiers = g["tier"].values
        for tier in sorted(set(tiers)):
            m = tiers == tier
            rec["tiers"][tier] = {"n": int(m.sum()), "hit_rate": round(float(hit[m].mean() * 100), 1),
                                  "pnl": round(float(pnl[m].sum()), 0), "pnl_per_hour": round(float(pnl[m].mean()), 2)}
        rated = np.isin(tiers, ["high", "medium"])
        rec["rated"] = {"n": int(rated.sum()), "hit_rate": round(float(hit[rated].mean() * 100), 1) if rated.any() else None,
                        "pnl": round(float(pnl[rated].sum()), 0)}
        watches = {"big_pos_watch": (g["dart"] > BIG_T).values, "big_neg_watch": (g["dart"] < -BIG_T).values,
                   "sbg_watch": (g["rt"] <= LOW_RT).values}
        rec["watch"] = {}
        for col, event in watches.items():
            if col not in g.columns:
                continue
            f = g[col].astype(bool).values
            rec["watch"][col] = {
                "n_flagged": int(f.sum()), "flagged_pct": round(float(f.mean() * 100), 1),
                "precision": round(float(event[f].mean() * 100), 1) if f.any() else None,
                "recall": round(float((event & f).sum() / event.sum() * 100), 1) if event.sum() else None,
                "base_rate": round(float(event.mean() * 100), 1), "n_events": int(event.sum()),
            }
        bins = pd.cut(g["p_pos"], [0, .5, .6, .7, .8, .9, 1.0])
        rec["calibration_p_pos"] = [
            {"bin": str(b), "n": int(len(x)), "p_mean": round(float(x["p_pos"].mean()), 3),
             "observed": round(float((x["dart"] > 0).mean()), 3), "e_dart": round(float(x["dart"].mean()), 2)}
            for b, x in g.groupby(bins, observed=True)]
        out.setdefault(vintage, {})[source] = rec
    return out


def build(directory, zone=ZONE):
    actuals = _actuals(zone)
    return {"zone": zone, "directory": str(directory), "prices": score_prices(directory, actuals, zone),
            "signal": score_signal(directory, actuals, zone)}


def format_card(card, label):
    lines = [f"--- {label} ---"]
    for prefix, vintages in card["prices"].items():
        for vintage, r in vintages.items():
            extra = ""
            if prefix == "rtm":
                extra = (f" | RT=DAMpred ${r.get('baseline_rt_eq_dam_pred_mae')} RT=DAMreal ${r.get('baseline_rt_eq_dam_real_mae')}"
                         f" | tail<=5 pred {r['tail_low_predicted_pct']}% vs real {r['tail_low_actual_pct']}%"
                         f" | tail>150 pred {r['tail_high_predicted_pct']}% vs real {r['tail_high_actual_pct']}%")
            if prefix == "spread":
                extra = (f" | sign {r['sign_hit_rate']}% (base {r['base_rate_pos']}%)"
                         f" pnl ${r['pnl_follow_sign']:.0f} vs always-gen ${r['pnl_always_gen']:.0f}")
                if "derived_mae" in r:
                    extra += f" | derived DAM-RT: MAE ${r['derived_mae']} sign {r['derived_sign_hit_rate']}% pnl ${r['derived_pnl_follow_sign']:.0f}"
            band = f" | P10-P90 cov {r['band_p10_p90_coverage']}% width ${r['band_width_mean']}" if "band_p10_p90_coverage" in r else ""
            lines.append(f"{prefix:6s} {vintage:8s} {r['n_days']:3d}d  MAE ${r['mae']:6.2f}  RMSE ${r['rmse']:6.2f}"
                         f"  bias ${r['bias']:+6.2f}  analog ${r['analog_mae']}{band}{extra}")
    for vintage, sources in (card.get("signal") or {}).items():
        for source, r in sources.items():
            tiers = "  ".join(f"{k}: n={v['n']} hit={v['hit_rate']}% pnl=${v['pnl']:.0f}" for k, v in r["tiers"].items())
            lines.append(f"signal {vintage:8s} {source:10s} {r['n_days']:3d}d  rated n={r['rated']['n']} hit={r['rated']['hit_rate']}%"
                         f" pnl=${r['rated']['pnl']:.0f} (always-gen ${r['pnl_always_gen']:.0f}, optimal ${r['pnl_optimal']:.0f}) | {tiers}")
            for k, w in r["watch"].items():
                lines.append(f"       {k}: flagged {w['flagged_pct']}%  precision {w['precision']}%  recall {w['recall']}%"
                             f"  (base {w['base_rate']}%, {w['n_events']} events)")
    return "\n".join(lines)


def default_json_path(zone=ZONE):
    return DATA_DIR / f"forecast_scorecard{zone_suffix(zone)}.json"


def save(card, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser(description="Score archived forecasts against realized prices.")
    ap.add_argument("--dir", default=str(DATA_DIR), help="folder holding *_forecast_history.csv / spread_signal_history.csv")
    ap.add_argument("--compare", default=None, help="second folder (candidate) to print next to --dir")
    ap.add_argument("--json", default=None, help="where to write the JSON (default: data/forecast_scorecard{_ZONE}.json when scoring data/)")
    ap.add_argument("--zone", default=ZONE)
    args = ap.parse_args()

    card = build(args.dir, args.zone)
    print(format_card(card, f"baseline: {args.dir} [{args.zone}]"))
    if args.compare:
        print(format_card(build(args.compare, args.zone), f"candidate: {args.compare} [{args.zone}]"))
    json_path = args.json or (default_json_path(args.zone) if Path(args.dir).resolve() == DATA_DIR.resolve() else None)
    if json_path:
        save(card, json_path)


if __name__ == "__main__":
    main()
