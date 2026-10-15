import numpy as np
import pandas as pd

from app.core import FEATURES, INTERVAL_MS
from app.portfolio import load_config
from app.single_market_replay import select_policy, sequential_scores, threshold_candidates


def test_candidates_match_validation_score_scale_with_exit_below_entry():
    candidates = threshold_candidates(np.linspace(.02, .2, 100), load_config())
    assert {'entry': .65, 'exit': .4} in candidates
    adaptive = [item for item in candidates if item['entry'] < .65]
    assert len(adaptive) == 4
    assert all(0 < item['exit'] < item['entry'] < .21 for item in adaptive)


def test_flat_predictions_do_not_create_a_forced_lower_threshold():
    assert threshold_candidates([.1] * 100, load_config()) == [{'entry': .65, 'exit': .4}]


def test_policy_selection_requires_profitable_validation_with_enough_trades():
    def candidate(count, profit, halted=False):
        return {'validation': {'closed_trades': count, 'net_return': profit,
                                'halted': halted, 'max_drawdown': -.1}}
    rejected = [candidate(0, 0), candidate(19, .5), candidate(20, -.01), candidate(30, .2, True)]
    assert select_policy(rejected) is None
    accepted = candidate(25, .01)
    assert select_policy([*rejected, accepted]) == accepted


def test_future_outcomes_cannot_change_predictions_before_they_mature():
    rows = pd.DataFrame({key: [0.] * 10 for key in FEATURES})
    rows['open_time'] = np.arange(10) * INTERVAL_MS
    rows['target'] = 0
    changed = rows.copy()
    changed.loc[0, 'target'] = 1
    baseline = np.full(10, .3)
    original_scores = sequential_scores(rows, baseline, 4)
    changed_scores = sequential_scores(changed, baseline, 4)
    np.testing.assert_equal(original_scores[:4], changed_scores[:4])
    assert changed_scores[4] > original_scores[4]
