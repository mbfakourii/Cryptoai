import copy
import json

import pytest

from app.core import INTERVAL_MS
from app.online_learning import new_learning_state
from app.online_policy import evaluate_online_policy
from app.portfolio import load_config, new_wallet, paper_cycle


def setup_policy():
    learning = new_learning_state(4)
    learning['market'].update(samples=5000, evaluated=5000, evaluation_batches=96,
                              brier=.1, baseline_brier=.2)
    config = load_config()
    markets = {'AAAUSDT': {'symbol': 'AAAUSDT', 'step_size': '.001',
                           'min_qty': '.001', 'max_qty': '10000', 'min_notional': 5.}}
    return learning, config, markets


def cycle(learning, config, markets, candle, price=100., **overrides):
    signal = {'symbol': 'AAAUSDT', 'action': 'HOLD', 'model_score': .8,
              'shadow_model_score': .8, 'entry_allowed': False,
              'feedback_entry_allowed': True, 'atr': 1., 'horizon_bars': 4,
              'candle_time': candle * INTERVAL_MS}
    signal.update(overrides.pop('signal', {}))
    return evaluate_online_policy(
        learning, {'AAAUSDT': signal}, {'AAAUSDT': {'bid': price, 'ask': price + .01}},
        markets, overrides.pop('errors', {}), overrides.pop('ready', True),
        (candle + 1) * INTERVAL_MS, config, overrides.pop('revision', 'baseline-v1'),
        overrides.pop('baseline_approved', False))


def profitable_history():
    learning, config, markets = setup_policy()
    for index in range(20):
        cycle(learning, config, markets, 2 * index, 100.)
        cycle(learning, config, markets, 2 * index + 1, 104.)
        learning = json.loads(json.dumps(learning))
    return learning, config, markets


def test_forward_shadow_profits_enable_actual_paper_entry_without_baseline_retraining():
    learning, config, markets = profitable_history()
    decisions, summary = cycle(learning, config, markets, 40)
    assert summary['approved'] and summary['recent_closed_trades'] == 20
    assert summary['recent_net_pnl'] > 0
    assert decisions['AAAUSDT']['action'] == 'BUY'
    wallet, events = paper_cycle(new_wallet(config), decisions,
                                 {'AAAUSDT': {'bid': 100., 'ask': 100.01}}, markets,
                                 {}, 41 * INTERVAL_MS, config)
    assert len(events) == 1 and events[0]['side'] == 'BUY'
    assert wallet['closed_count'] == 0


def test_one_profitable_trade_or_better_brier_alone_cannot_enable_entries():
    learning, config, markets = setup_policy()
    decisions, summary = cycle(learning, config, markets, 0)
    assert not summary['approved'] and decisions['AAAUSDT']['action'] == 'HOLD'
    _, summary = cycle(learning, config, markets, 1, 104.)
    assert summary['recent_net_pnl'] > 0 and not summary['approved']


def test_small_gross_gains_are_rejected_after_fees_spread_and_slippage():
    learning, config, markets = setup_policy()
    for index in range(20):
        cycle(learning, config, markets, index * 5, 100.)
        _, summary = cycle(learning, config, markets, index * 5 + 4, 100.1)
    assert summary['closed_trades'] == 20
    assert summary['realized_net_pnl'] < 0
    assert not summary['approved']


@pytest.mark.parametrize('failure', ['brier', 'candles', 'context', 'errors', 'halted'])
def test_approval_is_revoked_when_a_required_condition_fails(failure):
    learning, config, markets = profitable_history()
    kwargs = {}
    if failure == 'brier':
        learning['market']['brier'] = .3
    elif failure == 'candles':
        learning['market']['evaluation_batches'] = 95
    elif failure == 'context':
        kwargs['ready'] = False
    elif failure == 'errors':
        kwargs['errors'] = {'AAAUSDT': 'Unavailable'}
    else:
        learning['entry_policy']['wallet']['halted'] = True
    decisions, summary = cycle(learning, config, markets, 40, **kwargs)
    assert not summary['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'


def test_approval_does_not_override_trade_feedback_or_duplicate_candles():
    learning, config, markets = profitable_history()
    decisions, summary = cycle(learning, config, markets, 40,
                               signal={'feedback_entry_allowed': False})
    assert summary['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'
    decisions, _ = cycle(learning, config, markets, 41, signal={'action': 'WAITING'})
    assert decisions['AAAUSDT']['action'] == 'WAITING'


@pytest.mark.parametrize('change', ['revision', 'fees', 'threshold'])
def test_changed_policy_cannot_reuse_previous_profit_evidence(change):
    learning, config, markets = profitable_history()
    kwargs = {}
    if change == 'revision':
        kwargs['revision'] = 'baseline-v2'
    elif change == 'fees':
        config['fee_bps_per_side'] = 20
    else:
        config['entry_threshold'] = .7
    _, summary = cycle(learning, config, markets, 40, **kwargs)
    assert summary['closed_trades'] == 0 and not summary['approved']
    assert learning['market']['samples'] == 5000


def test_old_profits_expire_and_repeated_cycle_does_not_duplicate_shadow_trades():
    learning, config, markets = profitable_history()
    cycle(learning, config, markets, 40)
    original = copy.deepcopy(learning['entry_policy']['wallet'])
    cycle(learning, config, markets, 40)
    assert learning['entry_policy']['wallet'] == original
    _, summary = cycle(learning, config, markets, 40 + 8 * 96)
    assert summary['recent_closed_trades'] < 20 and not summary['approved']


def test_validated_baseline_entries_remain_available_during_shadow_warmup():
    learning, config, markets = setup_policy()
    decisions, summary = cycle(learning, config, markets, 0, baseline_approved=True,
                               signal={'action': 'BUY', 'entry_allowed': True})
    assert not summary['approved']
    assert decisions['AAAUSDT']['action'] == 'BUY'


def test_recent_losses_revoke_approval_even_with_positive_lifetime_profit():
    learning, config, markets = profitable_history()
    shadow = learning['entry_policy']['wallet']
    for event in shadow['trade_history']:
        if event['side'] == 'SELL':
            event['pnl'] = -.01
    decisions, summary = cycle(learning, config, markets, 40)
    assert summary['realized_net_pnl'] > 0
    assert summary['recent_net_pnl'] < 0
    assert not summary['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'


def test_approval_still_requires_the_current_signal_to_exceed_entry_threshold():
    learning, config, markets = profitable_history()
    decisions, summary = cycle(learning, config, markets, 40,
                               signal={'shadow_model_score': .5})
    assert summary['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'
