from datetime import date

import pytest

import check_forecasts as cfc

HEADER = "target_date,generated_at,hour,predicted_lmp,vintage,backfilled\n"


def _write(tmp_path, zone, rows):
    path = cfc.history_path(zone, tmp_path)
    path.write_text(HEADER + "".join(f"{d},2026-11-16T14:00:00+00:00,{h},1.0,{v},{b}\n" for d, h, v, b in rows))
    return path


def _all_zones(tmp_path, rows):
    for zone in cfc.ZONES:
        _write(tmp_path, zone, rows)


def test_history_path_suffix(tmp_path):
    assert cfc.history_path("OTTAWA", tmp_path).name == "spread_forecast_history.csv"
    assert cfc.history_path("NIAGARA", tmp_path).name == "spread_forecast_history_NIAGARA.csv"


def test_all_zones_live_passes(tmp_path):
    _all_zones(tmp_path, [("2026-11-18", h, "pre_dam", "False") for h in range(1, 25)])
    assert cfc.check("2026-11-18", "pre_dam", data_dir=tmp_path) == []


def test_backfilled_rows_do_not_count(tmp_path):
    # The 2026-11-04..15 hole: only walk-forward reconstructions for those days.
    _all_zones(tmp_path, [("2026-11-18", 1, "pre_dam", "True")])
    problems = cfc.check("2026-11-18", "pre_dam", data_dir=tmp_path)
    assert len(problems) == len(cfc.ZONES)
    assert all("no live pre_dam" in p for p in problems)


def test_other_vintage_or_day_does_not_count(tmp_path):
    _all_zones(tmp_path, [("2026-11-18", 1, "post_dam", "False"), ("2026-11-17", 1, "pre_dam", "False")])
    assert len(cfc.check("2026-11-18", "pre_dam", data_dir=tmp_path)) == len(cfc.ZONES)


def test_one_zone_missing_is_reported(tmp_path):
    _all_zones(tmp_path, [("2026-11-18", 1, "pre_dam", "False")])
    _write(tmp_path, "NIAGARA", [("2026-11-17", 1, "pre_dam", "False")])
    assert cfc.check("2026-11-18", "pre_dam", data_dir=tmp_path) == [
        "NIAGARA: no live pre_dam forecast for 2026-11-18"]


def test_missing_file_and_bad_columns(tmp_path):
    _write(tmp_path, "OTTAWA", [("2026-11-18", 1, "pre_dam", "False")])
    cfc.history_path("TORONTO", tmp_path).write_text("target_date,hour\n2026-11-18,1\n")
    problems = cfc.check("2026-11-18", "pre_dam", zones=("OTTAWA", "TORONTO", "NIAGARA"), data_dir=tmp_path)
    assert "missing expected columns" in problems[0] and problems[0].startswith("TORONTO")
    assert problems[1] == "NIAGARA: spread_forecast_history_NIAGARA.csv does not exist"


def test_rows_without_backfilled_column_count_as_live(tmp_path):
    path = cfc.history_path("OTTAWA", tmp_path)
    path.write_text("target_date,hour,vintage\n2026-11-18,1,pre_dam\n")
    assert cfc.check("2026-11-18", "pre_dam", zones=("OTTAWA",), data_dir=tmp_path) == []


def test_resolve_day():
    today = date(2026, 11, 30)
    assert cfc.resolve_day("today", today) == "2026-11-30"
    assert cfc.resolve_day("tomorrow", today) == "2026-12-01"
    assert cfc.resolve_day("2026-11-04", today) == "2026-11-04"
    with pytest.raises(ValueError):
        cfc.resolve_day("ayer", today)


def test_main_exits_1_when_missing(monkeypatch):
    monkeypatch.setattr(cfc, "check", lambda target, vintage: [f"OTTAWA: no live {vintage} forecast for {target}"])
    with pytest.raises(SystemExit) as exc:
        cfc.main(["--vintage", "pre_dam", "--day", "2026-11-18"])
    assert "Missing live pre_dam forecasts for 2026-11-18" in str(exc.value)
