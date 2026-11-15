import datetime as dt

import numpy as np
import pandas as pd
import pytest

import forecast_common as fc


def test_zone_helpers_and_forecast_file():
    assert fc.zone_suffix("OTTAWA") == ""
    assert fc.zone_suffix("NIAGARA") == "_NIAGARA"

    assert fc.forecast_file("dam", "forecast", "OTTAWA") == "dam_forecast.csv"
    assert fc.forecast_file("rtm", "forecast_history", "NIAGARA") == "rtm_forecast_history_NIAGARA.csv"
    assert fc.forecast_file("spread", "signal_meta", "TORONTO") == "spread_signal_meta_TORONTO.json"

    base_cols = ["ontario", "wind_forecast"]
    assert fc.zone_feature_cols(base_cols, "OTTAWA") == base_cols
    nw_cols = fc.zone_feature_cols(base_cols, "NORTHWEST")
    assert nw_cols == base_cols + fc.ZONE_FEATURES["NORTHWEST"]
    # No duplicate additions if already present
    assert fc.zone_feature_cols(nw_cols, "NORTHWEST") == nw_cols


def test_determine_target_date_and_errors():
    dam = pd.DataFrame(
        {"interval_start_local": pd.to_datetime(["2026-10-10 00:00:00-05:00", "2026-10-10 23:00:00-05:00"])}
    )
    today = dt.date(2026, 10, 10)
    assert fc.determine_target_date(dam, vintage="pre_dam", today=today) == dt.date(2026, 10, 11)

    # Once tomorrow's DAM is in, pre_dam raises with dam_already_out=True, while post_dam succeeds.
    dam_next = pd.DataFrame(
        {"interval_start_local": pd.to_datetime(["2026-10-11 23:00:00-05:00"])}
    )
    assert fc.determine_target_date(dam_next, vintage="post_dam", today=today) == dt.date(2026, 10, 11)
    with pytest.raises(fc.TargetDateError) as exc_info:
        fc.determine_target_date(dam_next, vintage="pre_dam", today=today)
    assert exc_info.value.dam_already_out is True

    # Behind DAM raises with dam_already_out=False
    dam_old = pd.DataFrame(
        {"interval_start_local": pd.to_datetime(["2026-10-08 23:00:00-05:00"])}
    )
    with pytest.raises(fc.TargetDateError) as exc_old:
        fc.determine_target_date(dam_old, vintage="pre_dam", today=today)
    assert exc_old.value.dam_already_out is False


def test_information_cutoff_and_sunset_hour():
    tz = "Etc/GMT+5"
    target = dt.date(2026, 10, 11)
    assert fc.information_cutoff(target, "pre_dam", tz) == pd.Timestamp("2026-10-10 07:00:00", tz=tz)
    assert fc.information_cutoff(target, "post_dam", tz) == pd.Timestamp("2026-10-10 13:00:00", tz=tz)

    summer_sunset, winter_sunset = fc.sunset_hour(["2026-06-21", "2026-12-21"])
    assert summer_sunset > winter_sunset
    assert 15.5 < winter_sunset < 17.5
    assert 19.0 < summer_sunset < 21.0


def test_build_grid_calendar_holidays_and_ramps(tmp_path, monkeypatch):
    # Ensure build_grid never reads real data/forecast_inputs_dayahead.csv
    monkeypatch.setattr(fc, "DAYAHEAD_PATH", tmp_path / "missing_dayahead.csv")

    tz = "Etc/GMT+5"
    # Build 2 days of history (2026-09-06 Sunday, 2026-09-07 Ontario Labour Day) + target 2026-09-08
    hist_idx = pd.date_range("2026-09-06 00:00:00", "2026-09-07 23:00:00", freq="h", tz=tz)
    full_idx = pd.date_range("2026-09-06 00:00:00", "2026-09-08 23:00:00", freq="h", tz=tz)

    target_df = pd.DataFrame({"interval_start_local": hist_idx, "lmp": np.arange(len(hist_idx), dtype=float)})
    load_fc = pd.DataFrame({"interval_start_local": full_idx, "ontario": 15000.0 + np.arange(len(full_idx)) * 10.0})
    wind_fc = pd.DataFrame({"interval_start_local": full_idx, "wind_forecast": 500.0})
    weather = pd.DataFrame({"interval_start_local": full_idx, "temperature_2m": 20.0})
    supply = pd.DataFrame(
        {
            "interval_start_local": full_idx,
            "net_load": 8000.0 + np.arange(len(full_idx)) * 5.0,
            "solar_forecast": np.where((full_idx.hour >= 8) & (full_idx.hour <= 18), 300.0, 0.0),
        }
    )

    df = fc.build_grid(target_df, load_fc, wind_fc, weather, dt.date(2026, 9, 8), tz, supply=supply)

    assert len(df) == 72
    # Target day has NaN lmp
    target_rows = df[df["interval_start_local"].dt.date == dt.date(2026, 9, 8)]
    assert len(target_rows) == 24
    assert target_rows["lmp"].isna().all()

    # Weekend & Ontario statutory holiday detection
    sun_row = df[df["interval_start_local"].dt.date == dt.date(2026, 9, 6)].iloc[0]
    labour_day_row = df[df["interval_start_local"].dt.date == dt.date(2026, 9, 7)].iloc[0]
    assert sun_row["is_weekend"] == 1
    assert labour_day_row["is_weekend"] == 0
    assert labour_day_row["is_holiday"] == 1
    assert target_rows.iloc[0]["is_holiday"] == 0

    # Ramp columns present and final hour ramp_out falls back cleanly without NaN
    for col in fc.RAMP_COLS:
        assert col in df.columns
    assert df["load_ramp_out"].iloc[-1] == pytest.approx(df["load_ramp_in"].iloc[-1])
    assert df["solar_ramp_out"].iloc[-1] == pytest.approx(0.0)


def test_add_lag_features_causal_dam_vs_rtm():
    tz = "Etc/GMT+5"
    idx = pd.date_range("2026-10-01 00:00:00", "2026-10-10 23:00:00", freq="h", tz=tz)
    lmp = np.arange(len(idx), dtype=float)
    df = pd.DataFrame({"interval_start_local": idx, "lmp": lmp, "hour": idx.hour})

    # DAM: known_through_yesterday=True -> lag_1d is 24h prior
    dam_df = fc.add_lag_features(df.copy(), prefix="dam", known_through_yesterday=True)
    assert dam_df["dam_lag_1d"].iloc[24] == pytest.approx(lmp[0])
    assert dam_df["dam_lag_7d"].iloc[168] == pytest.approx(lmp[0])
    assert dam_df["dam_roll_7d"].iloc[24 * 4 - 1] == pytest.approx(np.mean(lmp[: 24 * 3]))

    # RTM: known_through_yesterday=False -> lag_2d is 48h prior; today_early_* only uses hours < RT_HOURS_KNOWN - 1
    rtm_df = fc.add_lag_features(df.copy(), prefix="rtm", known_through_yesterday=False, vintage="pre_dam")
    assert rtm_df["rtm_lag_2d"].iloc[48] == pytest.approx(lmp[0])

    # pre_dam RT_HOURS_KNOWN = 7 -> early_hours = 6 (hours 0..5 of day t-1)
    # On 2026-10-02 (index 24..47), prev_day is 2026-10-01 (lmp 0..23); hours 0..5 have lmp 0..5.
    day2 = rtm_df[rtm_df["interval_start_local"].dt.date == dt.date(2026, 10, 2)]
    assert np.allclose(day2["rtm_today_early_mean"], 2.5)
    assert np.allclose(day2["rtm_today_early_max"], 5.0)


def test_pure_scoring_and_utility_helpers():
    # usable_feature_cols
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, np.nan], "c": [0.0, 2.0]})
    assert fc.usable_feature_cols(df, ["a", "b", "c"]) == ["a", "c"]

    # training_target
    s = pd.DataFrame({"lmp": [-500.0, 25.0, 500.0]})
    assert fc.training_target(s, target_clip=100.0).tolist() == [-100.0, 25.0, 100.0]
    assert fc.training_target(s, target_clip=None).tolist() == [-500.0, 25.0, 500.0]

    # order_band ensures p10 <= predicted_lmp <= p90
    band_df = pd.DataFrame({"p10": [30.0, 5.0], "predicted_lmp": [20.0, 15.0], "p90": [10.0, 12.0]})
    ordered = fc.order_band(band_df)
    assert ordered["p10"].tolist() == [10.0, 5.0]
    assert ordered["p90"].tolist() == [30.0, 15.0]

    # directional_backtest
    assert fc.directional_backtest(None) is None
    db = fc.directional_backtest({"test_actual": [10.0, -20.0, 5.0], "test_predicted": [2.0, -3.0, -1.0]})
    assert db["n_hours"] == 3
    assert db["correct"] == 2
    assert db["total_pnl"] == pytest.approx(25.0)  # +10 +20 -5
    assert db["optimal_pnl"] == pytest.approx(35.0)

    # recommend_hour skips hours with missing features when complete alternatives exist
    metrics = {"hourly_mae": {1: 2.0, 2: 5.0, 3: 8.0}}
    df_target = pd.DataFrame({"hour": [0, 1, 2], "feat": [np.nan, 1.0, 2.0]})
    best_h, best_mae = fc.recommend_hour(metrics, df_target, ["feat"])
    assert best_h == 2
    assert best_mae == pytest.approx(5.0)
