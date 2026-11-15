import sys
import types

import pandas as pd
import pytest

# gridstatusio is an optional runtime dep in ingest scripts; stub it if absent before importing update_common.
if "gridstatusio" not in sys.modules:
    stub = types.ModuleType("gridstatusio")

    class _PlaceholderClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

    stub.GridStatusClient = _PlaceholderClient
    sys.modules["gridstatusio"] = stub

import update_common  # noqa: E402


def _make_fake_client(new_df, calls_log=None):
    class FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key

        def get_dataset(self, **kwargs):
            if calls_log is not None:
                calls_log.append(kwargs)
            return new_df.copy()

    return FakeClient


def test_fetch_and_merge_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GRIDSTATUS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GRIDSTATUS_API_KEY"):
        update_common.fetch_and_merge(
            dataset_id="ieso_dam",
            data_path=tmp_path / "out.csv",
            dedup_subset=["interval_start_local"],
            sort_by=["interval_start_local"],
            past_hours=24,
            forecast_hours=24,
            keep_cols=["interval_start_local", "lmp"],
        )


def test_fetch_and_merge_incremental_dedup_and_column_trimming(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDSTATUS_API_KEY", "test-key")
    out_path = tmp_path / "dam.csv"

    # Existing CSV has a legacy extra column ("extra_col") that should be trimmed by keep_cols.
    existing = pd.DataFrame(
        [
            {"interval_start_local": "2026-10-10 00:00:00-05:00", "location": "OTTAWA", "lmp": 20.0, "extra_col": "old"},
            {"interval_start_local": "2026-10-10 01:00:00-05:00", "location": "OTTAWA", "lmp": 25.0, "extra_col": "old"},
        ]
    )
    existing.to_csv(out_path, index=False)

    # API returns an updated value for 01:00 plus a new 02:00 row, out of order, with extra API columns.
    new_df = pd.DataFrame(
        [
            {"interval_start_local": "2026-10-10 02:00:00-05:00", "location": "OTTAWA", "lmp": 40.0, "publish_time": "x"},
            {"interval_start_local": "2026-10-10 01:00:00-05:00", "location": "OTTAWA", "lmp": 99.0, "publish_time": "x"},
        ]
    )
    calls = []
    monkeypatch.setattr(update_common, "GridStatusClient", _make_fake_client(new_df, calls))

    result = update_common.fetch_and_merge(
        dataset_id="ieso_lmp_day_ahead_hourly",
        data_path=out_path,
        dedup_subset=["interval_start_local", "location"],
        sort_by=["interval_start_local", "location"],
        past_hours=48,
        forecast_hours=24,
        keep_cols=["interval_start_local", "location", "lmp"],
    )

    assert len(calls) == 1
    assert calls[0]["publish_time"] == "latest"
    assert list(result.columns) == ["interval_start_local", "location", "lmp"]
    assert len(result) == 3
    # Sorted ascending and keep="last" overwrote 01:00 from 25.0 -> 99.0
    assert result["lmp"].tolist() == [20.0, 99.0, 40.0]

    saved = pd.read_csv(out_path)
    assert list(saved.columns) == ["interval_start_local", "location", "lmp"]
    assert saved["lmp"].tolist() == [20.0, 99.0, 40.0]


def test_fetch_and_merge_missing_required_column_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDSTATUS_API_KEY", "test-key")
    bad_df = pd.DataFrame([{"interval_start_local": "2026-10-10 00:00:00-05:00"}])
    monkeypatch.setattr(update_common, "GridStatusClient", _make_fake_client(bad_df))

    with pytest.raises(RuntimeError, match="missing expected columns"):
        update_common.fetch_and_merge(
            dataset_id="ieso_dam",
            data_path=tmp_path / "out.csv",
            dedup_subset=["interval_start_local"],
            sort_by=["interval_start_local"],
            past_hours=24,
            forecast_hours=0,
            keep_cols=["interval_start_local", "lmp"],
        )


def test_fetch_and_merge_keep_cols_none_preserves_all_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDSTATUS_API_KEY", "test-key")
    out_path = tmp_path / "adequacy.csv"
    new_df = pd.DataFrame(
        [
            {"interval_start_local": "2026-10-10 01:00:00-05:00", "nuclear": 9000, "wind": 1200},
            {"interval_start_local": "2026-10-10 00:00:00-05:00", "nuclear": 9100, "wind": 1100},
        ]
    )
    monkeypatch.setattr(update_common, "GridStatusClient", _make_fake_client(new_df))

    result = update_common.fetch_and_merge(
        dataset_id="ieso_adequacy",
        data_path=out_path,
        dedup_subset=["interval_start_local"],
        sort_by=["interval_start_local"],
        past_hours=24,
        forecast_hours=24,
        keep_cols=None,
    )

    assert list(result.columns) == ["interval_start_local", "nuclear", "wind"]
    assert result["nuclear"].tolist() == [9100, 9000]
    assert out_path.exists()
