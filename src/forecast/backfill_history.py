"""One-off: rebuild data/*_forecast_history.csv from git history. Every past run committed
{prefix}_forecast.csv + meta, so the archive can be reconstructed from those blobs.
Run from the repo root; safe to re-run (archive_forecast keeps the latest vintage per day)."""
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_common import DATA_DIR, archive_forecast  # noqa: E402


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def main(prefixes=("dam", "rtm", "spread")):
    for prefix in prefixes:
        csv_rel, meta_rel = f"data/{prefix}_forecast.csv", f"data/{prefix}_forecast_meta.json"
        (DATA_DIR / f"{prefix}_forecast_history.csv").unlink(missing_ok=True)
        shas = git("log", "--reverse", "--format=%H", "--", csv_rel).split()
        for sha in shas:  # oldest first, so a later vintage for the same day wins
            try:
                out = pd.read_csv(io.StringIO(git("show", f"{sha}:{csv_rel}")))
                meta = json.loads(git("show", f"{sha}:{meta_rel}"))
            except subprocess.CalledProcessError:
                continue
            if "analog_lmp_2" not in out.columns:
                out["analog_lmp_2"] = float("nan")
            archive_forecast(prefix, out, meta)


if __name__ == "__main__":
    main()
