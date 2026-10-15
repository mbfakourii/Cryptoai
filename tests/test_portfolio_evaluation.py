import numpy as np
import pandas as pd
import pytest

from app.core import INTERVAL_MS
from app.portfolio import load_config
from app.portfolio_evaluation import EVALUATION_POLICY, evaluation_passed, shared_wallet_results
from app.portfolio_models import scheduled_trades, select_entry_policy


def candles(symbol='AAAUSDT', count=120, growth=0.):
    return pd.DataFrame({'symbol': symbol, 'open_time': np.arange(count) * INTERVAL_MS,
                         'open': 100 + np.arange(count) * growth, 'atr': 1.})


def market(symbol='AAAUSDT', minimum=1.):
    return {'symbol': symbol, 'step_size': '.0001', 'min_qty': '.0001',
            'max_qty': '100000', 'min_notional': minimum}


def test_unaffordable_profitable_symbol_cannot_hide_shared_wallet_losses():
    rising, falling = candles(growth=2.), candles('BBBUSDT', growth=-.2)
    config = load_config()
    isolated = [trade['net_return'] for frame in (rising, falling)
                for trade in scheduled_trades(frame, np.full(len(frame), .45), .45, config)]
    assert np.mean(isolated) > 0
    frame = pd.concat([rising, falling], ignore_index=True)
    markets = [market(minimum=10.), market('BBBUSDT')]
    result = shared_wallet_results(frame, np.full(len(frame), .45), .45, config, markets)
    assert result['net_return'] < 0
    assert result['blocked_entries']['BELOW_MINIMUM_ORDER'] > 0
    assert result['trade_count'] >= 20
    assert not evaluation_passed(result)
    policy = select_entry_policy(frame, np.full(len(frame), .45), config, markets)
    assert policy['status'] == 'no_validated_threshold'


def test_shared_capital_and_position_limit_apply_across_simultaneous_signals():
    frames = [candles(symbol, count=6) for symbol in ('AAAUSDT', 'BBBUSDT')]
    frame = pd.concat(frames, ignore_index=True)
    scores = np.tile([.9, .5, .5, .5, .5, .5], 2)
    result = shared_wallet_results(frame, scores, .8, {**load_config(), 'max_positions': 1},
                                  [market('AAAUSDT'), market('BBBUSDT')])
    assert result['trade_count'] == 1
    assert result['blocked_entries']['POSITION_LIMIT'] == 1
    assert result['equity'] == pytest.approx(20 * (1 + result['net_return']))
    assert result['realized_pnl'] == pytest.approx(result['equity'] - 20)


def test_final_eligible_entry_drains_at_the_last_available_execution_quote():
    frame = candles(count=7)
    scores = np.array([.5, .9, .5, .5, .5, .5, .5])
    result = shared_wallet_results(frame, scores, .8, load_config(), [market()])
    assert result['trade_count'] == 1
    assert result['open_positions'] == 0 and result['completed']
    assert result['fees_paid'] > 0 and result['net_return'] < 0
    assert result['end_time'] == 6 * INTERVAL_MS


def test_late_signal_cannot_leave_a_position_open_at_the_evaluation_boundary():
    frame = candles(count=7)
    scores = np.array([.5, .5, .9, .5, .5, .5, .5])
    result = shared_wallet_results(frame, scores, .8, load_config(), [market()])
    assert result['trade_count'] == result['open_positions'] == 0
    assert result['fees_paid'] == 0 and result['equity'] == 20


@pytest.mark.parametrize('defect', ['gap', 'duplicate', 'short_market', 'shifted_market', 'missing_metadata'])
def test_incomplete_synchronized_markets_fail_before_evaluating_profitability(defect):
    first, second = candles(), candles('BBBUSDT')
    markets = [market(), market('BBBUSDT')]
    if defect == 'gap':
        second = second.drop(index=60)
    elif defect == 'duplicate':
        second = pd.concat([second, second.iloc[[60]]])
    elif defect == 'short_market':
        second = second.iloc[1:]
    elif defect == 'shifted_market':
        second['open_time'] += INTERVAL_MS
    else:
        markets = markets[:1]
    frame = pd.concat([first, second], ignore_index=True)
    with pytest.raises(ValueError, match='synchronized'):
        shared_wallet_results(frame, np.full(len(frame), .9), .8, load_config(), markets)


@pytest.mark.parametrize('invalid', [{'evaluation_policy': 'independent_markets'}, {'synchronized': False},
                                    {'completed': False}, {'open_positions': 1}, {'halted': True},
                                    {'trade_count': 19}, {'net_return': 0.}, {'net_return': -.01},
                                    {'net_return': float('nan')}, {'net_return': None}])
def test_profitability_requires_completed_wallet_evidence_and_sufficient_trades(invalid):
    result = {'evaluation_policy': EVALUATION_POLICY, 'synchronized': True, 'completed': True,
              'open_positions': 0, 'halted': False, 'trade_count': 20, 'net_return': .01,
              'mean_net_return': .05}
    assert evaluation_passed(result)
    assert not evaluation_passed({**result, **invalid})


def test_numpy_wallet_returns_produce_serializable_approval_flags():
    result = {'evaluation_policy': EVALUATION_POLICY, 'synchronized': True, 'completed': True,
              'open_positions': 0, 'halted': False, 'trade_count': 20, 'net_return': np.float64(.01)}
    assert evaluation_passed(result) is True
