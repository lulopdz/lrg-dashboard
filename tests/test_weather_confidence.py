import datetime as dt

import pandas as pd

import update_weather_confidence as uwc


def _fresh(days, value):
    ts = pd.date_range(f"{days[0]} 00:00", periods=24 * len(days), freq="h", tz="-05:00")
    return pd.DataFrame({"timestamp": ts, "temperature_2m_p10": value, "temperature_2m_p90": value + 2})


def test_archive_keeps_only_tomorrow_and_upserts(tmp_path):
    path = tmp_path / "dayahead.csv"
    today = dt.date(2026, 11, 17)
    out = uwc.archive_dayahead(_fresh(["2026-11-17", "2026-11-18", "2026-11-19"], 1.0), path, today)
    assert out["target_date"].unique().tolist() == ["2026-11-18"] and len(out) == 24

    # A later run the same day replaces tomorrow's rows instead of duplicating them.
    uwc.archive_dayahead(_fresh(["2026-11-18"], 5.0), path, today)
    saved = pd.read_csv(path)
    assert len(saved) == 24 and (saved["temperature_2m_p10"] == 5.0).all()

    # The next day adds its own tomorrow and keeps the earlier one.
    uwc.archive_dayahead(_fresh(["2026-11-19"], 3.0), path, dt.date(2026, 11, 18))
    assert pd.read_csv(path)["target_date"].unique().tolist() == ["2026-11-18", "2026-11-19"]


def test_archive_without_tomorrow_is_a_no_op(tmp_path):
    path = tmp_path / "dayahead.csv"
    assert uwc.archive_dayahead(_fresh(["2026-11-17"], 1.0), path, dt.date(2026, 11, 17)) is None
    assert not path.exists()
