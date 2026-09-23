"""The day's forecasts for every zone, in one process -- what daily.yml (pre_dam) and
forecast_pm.yml (post_dam) run.

    python src/forecast/run_forecasts.py                      # pre_dam: DAM, RT and spread per zone, then the scorecards
    python src/forecast/run_forecasts.py --vintage post_dam   # RT and spread with tomorrow's real DAM
    python src/forecast/run_forecasts.py --zones OTTAWA       # just the dashboard's zone

One process instead of nine (3 zones x 3 scripts): the input CSVs are parsed once
(forecast_common.read_input_csv) instead of once per zone and model.

OTTAWA is the dashboard's zone: if it fails, the run fails. The other zones are best-effort --
a failure there prints the traceback and a ::warning:: annotation for the Actions summary,
but OTTAWA's forecast still gets committed and published.

A pre_dam run started after tomorrow's DAM is out has nothing honest left to forecast (see
forecast_common.determine_target_date): it says so and exits 0, since the afternoon post_dam
run covers that day."""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_common import VINTAGES, ZONE, ZONES, TargetDateError  # noqa: E402
import predict_dam  # noqa: E402
import predict_rtm  # noqa: E402
import predict_spread  # noqa: E402
import scorecard  # noqa: E402


def run_zone(zone, vintage):
    if vintage == "pre_dam":  # the DAM model has no post_dam vintage: by then the DAM is real
        predict_dam.main(zone=zone)
    predict_rtm.main(vintage, zone=zone)
    predict_spread.main(vintage, zone=zone)


def main():
    ap = argparse.ArgumentParser(description="Run the day's forecasts for every zone.")
    ap.add_argument("--vintage", choices=VINTAGES, default="pre_dam")
    ap.add_argument("--zones", default=",".join(ZONES), help=f"comma list (default {','.join(ZONES)})")
    args = ap.parse_args()
    zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    zones.sort(key=lambda z: z != ZONE)  # the dashboard's zone first, so it never waits on the others

    failed = []
    for zone in zones:
        t0 = time.time()
        print(f"\n===== {zone} ({args.vintage}) =====", flush=True)
        try:
            run_zone(zone, args.vintage)
            scorecard.save(scorecard.build(scorecard.DATA_DIR, zone), scorecard.default_json_path(zone))
        except TargetDateError as e:
            if e.dam_already_out and args.vintage == "pre_dam":
                print(f"::notice::Skipping the pre_dam forecasts: {e}")
                return 0
            raise
        except Exception:
            if zone == ZONE:
                raise
            traceback.print_exc()
            print(f"::warning::{zone} {args.vintage} forecast failed; {ZONE} is unaffected", flush=True)
            failed.append(zone)
            continue
        print(f"{zone} done in {time.time() - t0:.0f}s", flush=True)

    if failed:
        print(f"Best-effort zones that failed: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
