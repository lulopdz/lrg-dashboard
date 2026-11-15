import pandas as pd
import pytest

import scorecard


def _synthetic_actuals():
    rows_dam = [
        {"target_date": "2026-10-01", "hour": 1, "lmp": 50.0},
        {"target_date": "2026-10-01", "hour": 2, "lmp": 40.0},
        {"target_date": "2026-10-02", "hour": 1, "lmp": 60.0},
        {"target_date": "2026-10-02", "hour": 2, "lmp": 30.0},
    ]
    rows_rtm = [
        {"target_date": "2026-10-01", "hour": 1, "lmp": 30.0},   # spread +20
        {"target_date": "2026-10-01", "hour": 2, "lmp": 2.0},    # spread +38 (low RT <= 5)
        {"target_date": "2026-10-02", "hour": 1, "lmp": 160.0},  # spread -100 (spike > 150, big neg < -40)
        {"target_date": "2026-10-02", "hour": 2, "lmp": 10.0},   # spread +20
    ]
    dam = pd.DataFrame(rows_dam)
    rtm = pd.DataFrame(rows_rtm)
    spread = dam.merge(rtm, on=["target_date", "hour"], suffixes=("_dam", "_rtm"))
    spread["lmp"] = spread["lmp_dam"] - spread["lmp_rtm"]
    return {"dam": dam, "rtm": rtm, "spread": spread[["target_date", "hour", "lmp"]]}


def test_score_prices_separates_live_and_backfilled(tmp_path):
    actuals = _synthetic_actuals()

    # Write DAM, RTM, Spread histories for NIAGARA (tests zone suffix handling too)
    dam_hist = pd.DataFrame(
        [
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 1, "predicted_lmp": 48.0, "p10": 40.0, "p90": 55.0, "analog_lmp": 45.0, "backfilled": True},
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 42.0, "p10": 35.0, "p90": 50.0, "analog_lmp": 38.0, "backfilled": True},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 1, "predicted_lmp": 55.0, "p10": 45.0, "p90": 58.0, "analog_lmp": 52.0, "backfilled": False},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 35.0, "p10": 25.0, "p90": 40.0, "analog_lmp": 32.0, "backfilled": False},
        ]
    )
    rtm_hist = pd.DataFrame(
        [
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 1, "predicted_lmp": 25.0, "analog_lmp": 28.0, "backfilled": True},
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 4.0, "analog_lmp": 10.0, "backfilled": True},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 1, "predicted_lmp": 155.0, "analog_lmp": 50.0, "backfilled": False},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 12.0, "analog_lmp": 14.0, "backfilled": False},
        ]
    )
    spread_hist = pd.DataFrame(
        [
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 1, "predicted_lmp": 15.0, "analog_lmp": 17.0, "backfilled": True},
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 20.0, "analog_lmp": 28.0, "backfilled": True},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 1, "predicted_lmp": -80.0, "analog_lmp": 2.0, "backfilled": False},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 18.0, "analog_lmp": 18.0, "backfilled": False},
        ]
    )

    dam_hist.to_csv(tmp_path / "dam_forecast_history_NIAGARA.csv", index=False)
    rtm_hist.to_csv(tmp_path / "rtm_forecast_history_NIAGARA.csv", index=False)
    spread_hist.to_csv(tmp_path / "spread_forecast_history_NIAGARA.csv", index=False)

    res = scorecard.score_prices(tmp_path, actuals, zone="NIAGARA")

    assert set(res["dam"].keys()) == {"pre_dam", "pre_dam_reconstructed"}
    # Live DAM: errors (-5, +5) -> MAE 5.0; hour 1 actual 60 outside [45, 58], hour 2 actual 30 inside [25, 40] -> 50% coverage
    live_dam = res["dam"]["pre_dam"]
    assert live_dam["n_days"] == 1
    assert live_dam["n_hours"] == 2
    assert live_dam["mae"] == pytest.approx(5.0)
    assert live_dam["band_p10_p90_coverage"] == pytest.approx(50.0)

    # Reconstructed DAM: errors (-2, +2) -> MAE 2.0, 100% coverage
    reb_dam = res["dam"]["pre_dam_reconstructed"]
    assert reb_dam["mae"] == pytest.approx(2.0)
    assert reb_dam["band_p10_p90_coverage"] == pytest.approx(100.0)

    # RTM includes baselines & tail percentages
    live_rtm = res["rtm"]["pre_dam"]
    assert "baseline_rt_eq_dam_pred_mae" in live_rtm
    assert "baseline_rt_eq_dam_real_mae" in live_rtm
    assert live_rtm["tail_high_actual_pct"] == pytest.approx(50.0)
    assert live_rtm["tail_high_predicted_pct"] == pytest.approx(50.0)

    # Spread includes directional metrics & derived DAM-RT spread
    live_spread = res["spread"]["pre_dam"]
    assert live_spread["sign_hit_rate"] == pytest.approx(100.0)
    assert live_spread["pnl_follow_sign"] == pytest.approx(120.0)
    assert "derived_mae" in live_spread


def test_score_signal_and_strategy_frame_separation(tmp_path, monkeypatch):
    actuals = _synthetic_actuals()

    # Point DAYAHEAD_PATH and DATA_DIR to tmp_path so real data/ is never read
    dayahead_csv = tmp_path / "forecast_inputs_dayahead.csv"
    pd.DataFrame([{"target_date": "2026-10-01"}]).to_csv(dayahead_csv, index=False)
    monkeypatch.setattr(scorecard, "DAYAHEAD_PATH", dayahead_csv)
    monkeypatch.setattr(scorecard, "DATA_DIR", tmp_path)

    # Include spread_forecast_history without 'pick' in signal_history so with_picks computes picks
    spread_hist = pd.DataFrame(
        [
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 1, "predicted_lmp": 15.0, "backfilled": True},
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 20.0, "backfilled": True},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 1, "predicted_lmp": -80.0, "backfilled": False},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 18.0, "backfilled": False},
        ]
    )
    spread_hist.to_csv(tmp_path / "spread_forecast_history.csv", index=False)

    dam_hist = pd.DataFrame(
        [
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 1, "predicted_lmp": 48.0},
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 42.0},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 1, "predicted_lmp": 55.0},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 2, "predicted_lmp": 35.0},
        ]
    )
    dam_hist.to_csv(tmp_path / "dam_forecast_history.csv", index=False)

    sig_hist = pd.DataFrame(
        [
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 1, "p_pos": 0.70, "call": "DART > 0", "tier": "high", "big_pos_watch": False, "big_neg_watch": False, "sbg_watch": False, "backfilled": True},
            {"target_date": "2026-10-01", "vintage": "pre_dam", "hour": 2, "p_pos": 0.80, "call": "DART > 0", "tier": "high", "big_pos_watch": False, "big_neg_watch": False, "sbg_watch": True, "backfilled": True},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 1, "p_pos": 0.20, "call": "DART < 0", "tier": "high", "big_pos_watch": False, "big_neg_watch": True, "sbg_watch": False, "backfilled": False},
            {"target_date": "2026-10-02", "vintage": "pre_dam", "hour": 2, "p_pos": 0.65, "call": "DART > 0", "tier": "medium", "big_pos_watch": False, "big_neg_watch": False, "sbg_watch": False, "backfilled": False},
        ]
    )
    sig_hist.to_csv(tmp_path / "spread_signal_history.csv", index=False)

    sig_res = scorecard.score_signal(tmp_path, actuals, zone="OTTAWA")
    assert set(sig_res["pre_dam"].keys()) == {"live", "backfilled"}

    live = sig_res["pre_dam"]["live"]
    assert live["n_days"] == 1
    assert live["rated"]["hit_rate"] == pytest.approx(100.0)
    assert live["rated"]["pnl"] == pytest.approx(120.0)
    assert live["picks"]["n"] == 2
    assert live["picks"]["n_gen"] == 1
    assert live["picks"]["n_load"] == 1
    assert live["watch"]["big_neg_watch"]["precision"] == pytest.approx(100.0)
    assert live["watch"]["big_neg_watch"]["recall"] == pytest.approx(100.0)

    backfilled = sig_res["pre_dam"]["backfilled"]
    assert backfilled["n_days"] == 1
    assert backfilled["watch"]["sbg_watch"]["precision"] == pytest.approx(100.0)
    assert backfilled["watch"]["sbg_watch"]["recall"] == pytest.approx(100.0)

    frame = scorecard.strategy_frame(tmp_path, actuals, zone="OTTAWA")
    assert set(frame["segment"]) == {"live", "rebuilt_9am_inputs"}
    strat_scores = scorecard.score_strategies(frame)
    assert "pre_dam" in strat_scores
    assert "picks" in strat_scores["pre_dam"]["live"]
