import datetime as dt

import numpy as np
import pandas as pd
import pytest

from modeling import band_scale, calibrate_band

CFG = dict(window_days=30, min_days=10, target=0.8, k_min=1.0, k_max=2.5)
TARGET = dt.date(2026, 11, 18)


def _history_and_actual(n_days, miss_factor, raw_cols=True, start=dt.date(2026, 10, 19)):
    """Point 50, raw band 45-55 (half-width 5) every hour; the actual sits miss_factor half-widths
    away, alternating sides, so the k needed to cover every hour is exactly miss_factor."""
    rows, act = [], []
    tz = "-05:00"  # IESO prices carry a fixed EST offset (no DST)
    for d in range(n_days):
        day = start + dt.timedelta(days=d)
        for hour in range(1, 25):
            sign = 1 if hour % 2 else -1
            row = {"target_date": str(day), "vintage": "pre_dam", "hour": hour,
                   "predicted_lmp": 50.0, "p10": 45.0, "p90": 55.0}
            if raw_cols:
                row.update(p10_raw=45.0, p90_raw=55.0)
            rows.append(row)
            act.append({"interval_start_local": pd.Timestamp(f"{day} {hour - 1:02d}:00", tz=tz),
                        "lmp": 50.0 + sign * 5.0 * miss_factor})
    return pd.DataFrame(rows), pd.DataFrame(act)


def test_band_scale_recovers_needed_factor():
    hist, act = _history_and_actual(30, miss_factor=1.6)
    k, n_days = band_scale(hist, act, TARGET, "pre_dam", **CFG)
    assert n_days == 30
    assert k == pytest.approx(1.6)


def test_band_scale_needs_min_days_and_clips():
    hist, act = _history_and_actual(5, miss_factor=1.6, start=dt.date(2026, 11, 13))
    assert band_scale(hist, act, TARGET, "pre_dam", **CFG) == (None, 5)

    hist, act = _history_and_actual(30, miss_factor=0.3)  # band already too wide: never shrink
    assert band_scale(hist, act, TARGET, "pre_dam", **CFG)[0] == 1.0
    hist, act = _history_and_actual(30, miss_factor=9.0)
    assert band_scale(hist, act, TARGET, "pre_dam", **CFG)[0] == 2.5


def test_band_scale_ignores_target_day_and_later():
    hist, act = _history_and_actual(30, miss_factor=1.6)
    late_hist, late_act = _history_and_actual(10, miss_factor=2.4, start=TARGET)
    k, n_days = band_scale(pd.concat([hist, late_hist]), pd.concat([act, late_act]), TARGET, "pre_dam", **CFG)
    assert n_days == 30
    assert k == pytest.approx(1.6)


def test_band_scale_learns_from_raw_band_not_published_one():
    hist, act = _history_and_actual(30, miss_factor=1.6)
    hist = hist.assign(p10=42.0, p90=58.0)  # published (already widened) band must be ignored
    assert band_scale(hist, act, TARGET, "pre_dam", **CFG)[0] == pytest.approx(1.6)
    # Archives from before calibration have no *_raw columns: p10/p90 were the raw band.
    old, act = _history_and_actual(30, miss_factor=1.6, raw_cols=False)
    assert band_scale(old, act, TARGET, "pre_dam", **CFG)[0] == pytest.approx(1.6)


def test_calibrate_band_widens_around_point_and_keeps_raw():
    hist, act = _history_and_actual(30, miss_factor=2.0)
    target = pd.DataFrame({"predicted_lmp": [50.0, 80.0], "p10": [46.0, 70.0], "p90": [52.0, 100.0]})
    out, info = calibrate_band(target, hist, act, TARGET, "pre_dam", CFG)
    assert info["k"] == pytest.approx(2.0) and info["n_days"] == 30
    np.testing.assert_allclose(out["p10"], [42.0, 60.0])
    np.testing.assert_allclose(out["p90"], [54.0, 120.0])
    np.testing.assert_allclose(out["p10_raw"], [46.0, 70.0])
    assert (out["p10"] <= out["predicted_lmp"]).all() and (out["predicted_lmp"] <= out["p90"]).all()


def test_calibrate_band_without_history_publishes_raw():
    target = pd.DataFrame({"predicted_lmp": [50.0], "p10": [46.0], "p90": [52.0]})
    out, info = calibrate_band(target, None, pd.DataFrame(columns=["interval_start_local", "lmp"]),
                               TARGET, "pre_dam", CFG)
    assert info is None
    assert out["p10"].tolist() == [46.0] and out["p10_raw"].tolist() == [46.0]
