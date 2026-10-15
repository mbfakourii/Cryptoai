import numpy as np
import pandas as pd
import pytest

from app import btc_walk_forward
from app.btc_walk_forward import DAY_MS, choose_on_validation, replay_window, summarize, validate_history, windows
from app.core import INTERVAL_MS
from app.portfolio import load_config


def frame(rows=100):
    return pd.DataFrame({'open_time': np.arange(rows) * INTERVAL_MS,
                         'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 100., 'atr': 1.})


def market():
    return {'symbol': 'BTCUSDT', 'step_size': '.001', 'min_qty': '.001',
            'max_qty': '100000', 'min_notional': 5.}


def test_walk_forward_windows_use_only_prior_training_and_validation_with_disjoint_tests():
    folds = list(windows(0, 730 * DAY_MS))
    assert len(folds) == 9
    assert folds[0]['test_start'] == 210 * DAY_MS
    assert folds[-1]['test_end'] == 730 * DAY_MS
    for fold in folds:
        assert fold['train_start'] < fold['validation_start'] < fold['test_start'] < fold['test_end']
    assert all(left['test_end'] == right['test_start'] for left, right in zip(folds, folds[1:]))


@pytest.mark.parametrize('holding', [4, 8, 16])
def test_replay_drains_positions_and_charges_round_trip_costs(holding):
    raw = frame()
    policy = {'entry': .8, 'exit': .4, 'holding_bars': holding}
    config = load_config()
    result = replay_window(raw, np.full(100, .9), market(), 0, 100 * INTERVAL_MS, policy, config)
    assert result['closed_trades'] > 0
    assert result['fees_paid'] > 0
    assert result['net_return'] < 0
    free = replay_window(raw, np.full(100, .9), market(), 0, 100 * INTERVAL_MS, policy,
                         {**config, 'fee_bps_per_side': 0, 'slippage_bps_per_side': 0, 'backtest_spread_bps': 0})
    assert free['net_return'] == pytest.approx(0)


def test_replay_does_not_open_trades_that_could_cross_the_test_boundary():
    scores = np.full(100, .5)
    scores[-16:] = .9
    result = replay_window(frame(), scores, market(), 0, 100 * INTERVAL_MS,
                            {'entry': .8, 'exit': .4, 'holding_bars': 16}, load_config())
    assert result['closed_trades'] == 0
    assert result['net_return'] == 0


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'invalid_high'])
def test_history_rejects_missing_duplicate_and_invalid_candles(corruption):
    raw = frame()
    if corruption == 'missing':
        raw = raw.drop(index=50)
    elif corruption == 'duplicate':
        raw.loc[50, 'open_time'] = raw.loc[49, 'open_time']
    else:
        raw.loc[50, 'high'] = 90.
    with pytest.raises(ValueError, match='complete, ordered, valid'):
        validate_history(raw, 0, 100 * INTERVAL_MS)


def test_selection_cannot_use_profitable_test_to_override_failed_validation():
    rejected = {'validation': {'closed_trades': 2, 'net_return': .5, 'halted': False},
                'test': {'net_return': 1.}}
    accepted = {'validation': {'closed_trades': 20, 'net_return': .01, 'halted': False},
                'test': {'net_return': -.1}}
    assert choose_on_validation([rejected, accepted]) == (accepted, True)
    assert choose_on_validation([rejected]) == (rejected, False)


def test_failed_validation_means_cash_in_gated_summary_not_fabricated_profitable_trades():
    result = {'net_return': .1, 'closed_trades': 25, 'max_drawdown': -.01}
    folds = [{'strategies': {'model': {'validation_passed': False, 'test': result,
                                       'test_higher_costs': result}}}]
    summary = summarize(folds)['strategies']['model']
    assert summary['closed_trades'] == 25
    assert summary['normalized_compounded_return'] == pytest.approx(.1)
    assert summary['gated_closed_trades'] == 0
    assert summary['gated_normalized_compounded_return'] == 0


def test_history_download_retries_and_reuses_validated_cache(tmp_path, monkeypatch):
    calls = []

    def download(symbol, count, end):
        calls.append((symbol, count, end))
        if len(calls) == 1:
            raise RuntimeError('Temporary outage')
        return frame()

    monkeypatch.setattr(btc_walk_forward, 'candle_frame', download)
    monkeypatch.setattr(btc_walk_forward.time, 'sleep', lambda seconds: None)
    cache = tmp_path / 'btc.csv'
    first = btc_walk_forward.load_history(0, 100 * INTERVAL_MS, cache)
    second = btc_walk_forward.load_history(0, 100 * INTERVAL_MS, cache)
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 2
