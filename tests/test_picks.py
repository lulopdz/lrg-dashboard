import numpy as np

from picks import MIN_FC, N_PICKS, top_picks


def test_top_picks_ranking_and_n_cap():
    pred = [10.0, -30.0, 25.0, 8.0, -12.0, 6.0, 40.0]
    p_pos = [0.7, 0.2, 0.8, 0.6, 0.3, 0.9, 0.9]

    ranks = top_picks(pred, p_pos, n=3, min_fc=5.0)
    # Largest |pred| eligible: idx 6 (40 -> 1), idx 1 (-30 -> 2), idx 2 (25 -> 3)
    assert ranks.tolist() == [0, 2, 3, 0, 0, 0, 1]


def test_top_picks_threshold_and_nan():
    pred = [4.99, -4.99, 5.0, -5.0, np.nan]
    p_pos = [0.9, 0.1, 0.8, 0.2, 0.9]

    ranks = top_picks(pred, p_pos, n=N_PICKS, min_fc=MIN_FC)
    assert ranks[0] == 0
    assert ranks[1] == 0
    assert ranks[4] == 0
    assert set(ranks[[2, 3]]) == {1, 2}


def test_top_picks_sign_alignment_asymmetry():
    # Load hours (pred < 0) are vetoed unless p_pos < 0.5;
    # Gen hours (pred > 0) ignore classifier disagreement.
    pred = [-20.0, -15.0, -10.0, 12.0]
    p_pos = [0.75, 0.50, 0.30, 0.15]

    ranks = top_picks(pred, p_pos, n=5, min_fc=5.0)
    # -20 (p_pos=0.75) vetoed, -15 (p_pos=0.50 -> sign(0)=0 != -1) vetoed,
    # +12 (gen, p_pos=0.15) kept -> rank 1, -10 (load, p_pos=0.30) kept -> rank 2.
    assert ranks.tolist() == [0, 0, 2, 1]


def test_top_picks_all_ineligible_returns_zeros():
    pred = [1.0, -2.0, -10.0]
    p_pos = [0.9, 0.1, 0.8]
    ranks = top_picks(pred, p_pos, n=5, min_fc=5.0)
    assert (ranks == 0).all()
