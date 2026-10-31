"""The archive scored as trading strategies: what 1 MW of virtuals would have made following each
rule, day by day, with the risk numbers next to the P&L. This is the yardstick for changes to
the spread models (README, "Evaluar un cambio"): MAE and hit rate are secondary, because the
money is in a few hours -- the 5% largest |DART| hours carry 38% of all |DART| (OTTAWA,
2025-05..2026-10), and from 2026-09-24 always-gen lost $1,471 at OTTAWA, $704 ahead without its
10 worst hours, all of them HE17-19.

Every strategy maps an hour to a side: +1 virtual gen (earns DART), -1 virtual load (earns
-DART), 0 flat. The fixed baselines are the bar a model has to clear:

    picks           the day's picks as archived (picks.py): the DART forecast's sign, top 5 hours (3 before 2026-10-31)
    picks_5         the same rule with 5 picks a day, recomputed for every day
    top5_fc         the 5 hours with the largest |DART forecast|, on its sign, no filters
    follow_sign     the DART forecast's sign, every hour
    always_gen      gen every hour
    gen_ex_evening  gen every hour but HE17-19, where the autumn RT spikes sit
    gen_top3_dam    gen in the 3 hours with the highest DAM (pre_dam: the DAM forecast, post_dam: the real DAM)

picks_5 and top5_fc are the operating point: collateral allows 3 to 5 positions a day, and over
the full-history walk-forward (2025-07-19..2026-10-30, four zones) 5 a day made ~50% more than 3
at about the same $/h, so a change to the models is judged on those two first.

The P&L is resampled by day (bootstrap) for a 90% interval: hours within a day move together,
so resampling hours would claim far more certainty than ~40 days carry. compare() does the same
on the per-day difference between two runs over their common days."""
import numpy as np
import pandas as pd

from picks import top_picks

EVENING_HOURS = (17, 18, 19)  # hour-ending
N_BOOT = 2000
CI = (5, 95)


def _sides(g):
    fc = g["fc"].to_numpy(dtype=float)
    hour = g["hour"].to_numpy()
    dam_rank = g.groupby("target_date")["dam_ref"].rank(ascending=False, method="first").to_numpy()
    fc_rank = g["fc"].abs().groupby(g["target_date"]).rank(ascending=False, method="first").to_numpy()
    picks_5 = np.zeros(len(g))
    for idx in g.groupby("target_date").indices.values():
        picks_5[idx] = top_picks(fc[idx], g["p_pos"].to_numpy(dtype=float)[idx], n=5)
    return {
        "picks": np.where(g["pick"].to_numpy() > 0, np.sign(np.nan_to_num(fc)), 0.0),
        "picks_5": np.where(picks_5 > 0, np.sign(np.nan_to_num(fc)), 0.0),
        "top5_fc": np.where(fc_rank <= 5, np.sign(np.nan_to_num(fc)), 0.0),
        "follow_sign": np.sign(np.nan_to_num(fc)),
        "always_gen": np.ones(len(g)),
        "gen_ex_evening": np.where(np.isin(hour, EVENING_HOURS), 0.0, 1.0),
        "gen_top3_dam": np.where(dam_rank <= 3, 1.0, 0.0),
    }


STRATEGIES = ("picks", "picks_5", "top5_fc", "follow_sign", "always_gen", "gen_ex_evening", "gen_top3_dam")


def daily_pnl(frame):
    """{strategy: Series of P&L per target_date} for one record (vintage x source). Days where a
    strategy took no position count as $0, so every strategy is judged on the same days."""
    g = frame.sort_values(["target_date", "hour"])
    days = pd.Index(sorted(g["target_date"].unique()), name="target_date")
    dart = g["dart"].to_numpy(dtype=float)
    out = {}
    for name, side in _sides(g).items():
        pnl = pd.Series(side * dart, index=g.index)
        out[name] = pnl.groupby(g["target_date"]).sum().reindex(days, fill_value=0.0)
    return out


def _boot_mean(values, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, CI)
    return [round(float(lo), 2), round(float(hi), 2)]


def _max_drawdown(daily):
    equity = np.cumsum(daily)
    return float((np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity).max()) if len(daily) else 0.0


def score(frame):
    """{strategy: metrics} for one record. frame: one row per (target_date, hour) with dart, fc,
    p_pos, pick and dam_ref."""
    g = frame.sort_values(["target_date", "hour"]).reset_index(drop=True)
    sides = _sides(g)
    dart = g["dart"].to_numpy(dtype=float)
    out = {}
    for name, daily in daily_pnl(g).items():
        side = sides[name]
        traded = side != 0
        hourly = side[traded] * dart[traded]
        out[name] = {
            "n_days": int(len(daily)), "n_trades": int(traded.sum()),
            "pnl": round(float(daily.sum()), 0),
            "pnl_per_day": round(float(daily.mean()), 2),
            "pnl_per_day_ci90": _boot_mean(daily.values),
            "pnl_per_trade": round(float(hourly.mean()), 2) if traded.any() else None,
            "hit_rate": round(float((hourly > 0).mean() * 100), 1) if traded.any() else None,
            "worst_hour": round(float(hourly.min()), 0) if traded.any() else None,
            "worst_day": round(float(daily.min()), 0),
            "max_drawdown": round(_max_drawdown(daily.values), 0),
            "pct_days_positive": round(float((daily[daily != 0] > 0).mean() * 100), 1) if (daily != 0).any() else None,
            "pnl_by_month": {k: round(float(v), 0) for k, v in daily.groupby(daily.index.str[:7]).sum().items()},
        }
    return out


def compare(base_frame, cand_frame):
    """{strategy: paired difference} between two runs on the days both have. The candidate is
    better when its daily P&L minus the baseline's is above zero; ci90 brackets that mean."""
    days = sorted(set(base_frame["target_date"]) & set(cand_frame["target_date"]))
    if not days:
        return {}
    b = daily_pnl(base_frame[base_frame["target_date"].isin(days)])
    c = daily_pnl(cand_frame[cand_frame["target_date"].isin(days)])
    out = {}
    for name in STRATEGIES:
        diff = (c[name] - b[name]).values
        out[name] = {"n_days": len(days), "base_pnl": round(float(b[name].sum()), 0), "cand_pnl": round(float(c[name].sum()), 0),
                     "diff_per_day": round(float(diff.mean()), 2), "diff_per_day_ci90": _boot_mean(diff),
                     "days_better": int((diff > 0).sum()), "days_worse": int((diff < 0).sum())}
    return out


def format_scores(scores, label):
    lines = [f"  {label}"]
    for name, r in scores.items():
        ci = r["pnl_per_day_ci90"]
        lines.append(f"    {name:15s} {r['n_days']:3d}d {r['n_trades']:5d}h  P&L ${r['pnl']:7.0f}  ${r['pnl_per_day']:7.2f}/día"
                     f" [IC90 {ci[0] if ci else '-'}, {ci[1] if ci else '-'}]  ${r['pnl_per_trade']}/h  acierto {r['hit_rate']}%"
                     f"  peor hora ${r['worst_hour']}  peor día ${r['worst_day']:.0f}  drawdown ${r['max_drawdown']:.0f}")
    return "\n".join(lines)


def format_compare(diffs, label):
    lines = [f"  {label}"]
    for name, r in diffs.items():
        ci = r["diff_per_day_ci90"]
        lines.append(f"    {name:15s} {r['n_days']:3d}d  base ${r['base_pnl']:7.0f} -> cand ${r['cand_pnl']:7.0f}"
                     f"  Δ ${r['diff_per_day']:+7.2f}/día [IC90 {ci[0] if ci else '-'}, {ci[1] if ci else '-'}]"
                     f"  días mejor/peor {r['days_better']}/{r['days_worse']}")
    return "\n".join(lines)
