"""Walk-forward reconstruction: re-run the predictors for a range of past target days, each
seeing only what the live run of that day could see (history cut at the run's information
cutoff, DAM feature from the archived vintage), and archive the results in a folder that
scorecard.py can grade next to the real archive.

    python src/forecast/walkforward.py --out /tmp/cand                      # every day in the archive
    python src/forecast/walkforward.py --out /tmp/cand --models rtm,spread  # DAM feature from data/'s archive
    python src/forecast/walkforward.py --out /tmp/cand --jobs 4             # 4 processes, ~4x faster
    python src/forecast/scorecard.py --dir data --compare /tmp/cand

Caveat shared with the live backtests: the load/wind/weather inputs are the latest published
version, except for target days that data/forecast_inputs_dayahead.csv covers (vintages.py),
where the 9:00 view is used. So absolute numbers are somewhat optimistic; comparisons between
two reconstructions are fair -- both see the same inputs -- which is what this is for: change
the code, rebuild, compare.

--jobs N splits the days into N contiguous chunks, each rebuilt by its own process into
out/_part<i>/ (the history archives are rewritten per day, so processes can't share a
folder), then merges the parts' histories into out/. Each process gets cores/N threads."""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_common import DATA_DIR, VINTAGES, ZONE  # noqa: E402


def target_days(args):
    if args.since or args.until:
        hist = pd.read_csv(DATA_DIR / "rtm_forecast_history.csv") if (DATA_DIR / "rtm_forecast_history.csv").exists() else None
        days = sorted(hist["target_date"].unique()) if hist is not None else []
    else:
        days = sorted(pd.read_csv(DATA_DIR / "dam_forecast_history.csv")["target_date"].unique())
    days = [d for d in days if (not args.since or d >= args.since) and (not args.until or d <= args.until)]
    if args.days:
        days = days[-args.days:]
    return [pd.Timestamp(d).date() for d in days]


def rebuild(days, out, models, vintage, dam_source, zone, label=""):
    """Rebuild `days` into `out`, one day at a time. Imported here, not at module level, so a
    --jobs worker sets its thread count before scikit-learn loads."""
    import predict_dam, predict_rtm, predict_spread  # noqa: E401
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dam_dir = out if dam_source == "candidate" else DATA_DIR
    t0 = time.time()
    for i, day in enumerate(days, 1):
        if "dam" in models:
            predict_dam.main(vintage, day, out, backtest_enabled=False, zone=zone)
        if "rtm" in models:
            predict_rtm.main(vintage, day, out, dam_forecast_dir=dam_dir, backtest_enabled=False, zone=zone)
        if "spread" in models:
            predict_spread.main(vintage, day, out, dam_forecast_dir=dam_dir, backtest_enabled=False, zone=zone)
        print(f"{label}[{i}/{len(days)}] {day} done ({time.time() - t0:.0f}s)", flush=True)
    return str(out)


def _worker(threads, *args):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["LOKY_MAX_CPU_COUNT"] = str(threads)
    return rebuild(*args)


def merge_parts(parts, out):
    """Concatenate each *_history*.csv across the part folders into out/."""
    names = sorted({p.name for part in parts for p in Path(part).glob("*_history*.csv")})
    for name in names:
        frames = [pd.read_csv(Path(part) / name) for part in parts if (Path(part) / name).exists()]
        merged = pd.concat(frames).sort_values([c for c in ("target_date", "vintage", "hour") if c in frames[0].columns])
        merged.to_csv(Path(out) / name, index=False)
        print(f"Merged {name}: {merged['target_date'].nunique()} days")
    for part in parts:
        shutil.rmtree(part, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Rebuild past forecasts walk-forward into a folder.")
    ap.add_argument("--out", required=True, help="output folder (created); becomes --compare for scorecard.py")
    ap.add_argument("--models", default="dam,rtm,spread", help="comma list of dam,rtm,spread (spread includes the signal)")
    ap.add_argument("--vintage", choices=VINTAGES, default="pre_dam")
    ap.add_argument("--dam-source", choices=["archived", "candidate"], default="archived",
                    help="where RTM/Spread take the target day's DAM forecast: data/ (archived vintages) or --out (this run's DAM)")
    ap.add_argument("--days", type=int, default=None, help="only the last N target days")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="YYYY-MM-DD")
    ap.add_argument("--zone", default=ZONE)
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes (each rebuilds a contiguous chunk of days)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",")]
    days = target_days(args)
    print(f"Reconstructing {len(days)} days ({days[0]} -> {days[-1]}), models={models}, vintage={args.vintage}, "
          f"dam from {'--out' if args.dam_source == 'candidate' else DATA_DIR}, jobs={args.jobs}")
    t0 = time.time()
    if args.jobs <= 1:
        rebuild(days, out, models, args.vintage, args.dam_source, args.zone)
    else:
        jobs = min(args.jobs, len(days))
        size = -(-len(days) // jobs)
        chunks = [days[i:i + size] for i in range(0, len(days), size)]
        threads = max(1, (os.cpu_count() or 2) // len(chunks))
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(_worker, threads, chunk, out / f"_part{i}", models, args.vintage, args.dam_source,
                                   args.zone, f"(part {i}) ") for i, chunk in enumerate(chunks)]
            parts = [f.result() for f in futures]
        merge_parts(parts, out)
    print(f"Finished in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
