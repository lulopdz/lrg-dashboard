"""Each day's hours to trade first, from the DART point forecast and the P(DART > 0) classifier.
Numpy only, so the page builder (dashboard_data.py) and scorecard.py can rank archived days that
predate the `pick` column with the exact rule predict_spread.py now archives."""
import numpy as np

# The side is always the DART point forecast's sign (virtual gen
# when > 0, virtual load when < 0) -- the one rule that kept paying after the market turned
# around 2026-09-20 (the tiers and the big-hour watch, fit on the trailing 60 days, kept calling
# DART > 0 and lost). Of the hours where the P(DART > 0) classifier sides with the forecast, the
# N_PICKS with the largest |forecast|, and none under MIN_FC. Chosen 2026-10-17 among ten
# rankings, walk-forward pre_dam over 62 days (08-16..10-16): $9-14/h at 1 MW and right 68-73%
# of the time in all four zones, against $2-4/h trading every hour, and the only ranking positive
# in every zone after 09-20. MIN_FC drops the near-zero hours a gen-leaning classifier leaves to
# pick from on some days: the same total P&L on ~20% fewer hours (no floor: $8-11/h, 65-71%).
# Five, not three, since 2026-10-31: collateral allows 3 to 5 positions a day, and over the
# full-history walk-forward (2025-07-19..2026-10-30, four zones) the fourth and fifth picks
# added ~50% to the total at about the same $/h; ranked, so a day with less collateral trades
# the first three (strategies.py scores both).
N_PICKS = 5
MIN_FC = 5.0  # $/MWh


def top_picks(pred, p_pos, n=N_PICKS, min_fc=MIN_FC):
    """Rank (1..n) of each hour among the day's picks, 0 for the rest. pred: DART forecast per
    hour; p_pos: P(DART > 0) per hour. Hours with no forecast or a forecast under min_fc either
    way are out, and so are load hours (pred < 0) the classifier doesn't side with, so a day can
    have fewer than n picks, or none.

    The classifier's veto applies to load only since 2026-10-31. On gen hours it was turning down
    good trades: dropping it there made more in all four zones over the full-history walk-forward
    (2025-07-19..2026-10-30, 5 picks a day, +$0.8 to +$6.0 per day; OTTAWA's 90% interval clear
    of zero). Load picks keep it: they pay ~$1.3/h against gen's ~$7.2/h, and none of the
    stricter load rules tried (|forecast|, P(DART > 0) or P(DART < -40) thresholds) beat it in
    three of four zones."""
    pred, p_pos = np.asarray(pred, dtype=float), np.asarray(p_pos, dtype=float)
    agree = np.sign(pred) == np.sign(p_pos - 0.5)
    eligible = (np.abs(np.nan_to_num(pred)) >= min_fc) & ((pred > 0) | agree)
    order = [i for i in np.argsort(-np.abs(np.nan_to_num(pred))) if eligible[i]][:n]
    rank = np.zeros(len(pred), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    return rank
