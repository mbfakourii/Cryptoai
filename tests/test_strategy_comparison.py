import numpy as np
import pandas as pd
import pytest

from app.core import INTERVAL_MS
from app.portfolio import load_config
from app.portfolio_models import scheduled_trades
from app.strategy_comparison import replay, trend_scores
from app import strategy_comparison


def candles(count=40):
    return pd.DataFrame({'open_time': np.arange(count) * INTERVAL_MS, 'open': 100.,
                         'close': 100., 'high': 101., 'low': 99., 'volume': 100., 'atr': 1.})


def test_threshold_backtest_exits_on_the_next_observed_quote_when_model_reverses():
    frame = candles(8)
    frame.loc[2, 'open'] = 99.9
    scores = np.array([.8, .2, .5, .5, .5, .5, .5, .5])
    trades = scheduled_trades(frame, scores, .7, load_config())
    assert len(trades) == 1
    assert trades[0]['result'] == 'MODEL'
    assert trades[0]['exit_time'] == 2 * INTERVAL_MS
    assert trades[0]['net_return'] < 0


def test_threshold_backtest_does_not_assume_fills_at_intrabar_stop_or_target():
    frame = candles(8)
    frame['high'] = 150.
    frame['low'] = 1.
    scores = np.array([.8] + [.5] * 7)
    trades = scheduled_trades(frame, scores, .7, load_config())
    assert trades[0]['result'] == 'TIME'
    assert trades[0]['exit_time'] - trades[0]['entry_time'] == 3600000
    assert trades[0]['net_return'] < 0


def test_stop_gap_is_filled_at_observed_bid_instead_of_the_stop_level():
    frame = candles(8)
    frame.loc[2, 'open'] = 80.
    trades = scheduled_trades(frame, np.array([.8] + [.5] * 7), .7, load_config())
    assert trades[0]['result'] == 'SL'
    assert trades[0]['net_return'] < -.2


def test_breakout_uses_previous_highs_and_never_future_candles():
    frame = candles()
    frame.loc[30, ['close', 'high', 'volume']] = [105., 106., 200.]
    scores = trend_scores(frame)
    assert scores[30] == .8
    changed = frame.copy()
    changed.loc[31:, ['close', 'high', 'volume']] = 10000.
    assert np.array_equal(trend_scores(changed)[:31], scores[:31])


def test_shared_wallet_replay_accounts_for_costs_and_model_exits():
    frame = candles()
    scores = np.full(len(frame), .5)
    scores[25], scores[26] = .8, .2
    market = {'symbol': 'AAAUSDT', 'step_size': '.001', 'min_qty': '.001',
              'max_qty': '100000', 'min_notional': 1.}
    result = replay({'AAAUSDT': frame}, {'AAAUSDT': scores}, [market],
                    25 * INTERVAL_MS, 35 * INTERVAL_MS, load_config(), .7)
    assert result['closed_trades'] == 1 and result['open_positions'] == 0
    assert result['net_return'] < 0 and result['fees_paid'] > 0
    assert result['equity'] == pytest.approx(20 * (1 + result['net_return']))


def test_strategy_selection_does_not_switch_to_the_best_test_result(monkeypatch):
    from app.core import FEATURES, make_features

    frame = candles()
    featured = make_features(frame).fillna(0)
    featured['target'] = np.arange(len(frame)) % 2
    monkeypatch.setattr(strategy_comparison, 'pooled_splits', lambda *args: (featured, featured, featured))

    class Model:
        def fit(self, *args, **kwargs):
            pass

        def predict_proba(self, frame):
            assert list(frame.columns) == FEATURES
            return np.tile([.5, .5], (len(frame), 1))

    monkeypatch.setattr(strategy_comparison, 'XGBClassifier', lambda **kwargs: Model())
    def select_policy(validation, scores, config, markets):
        assert markets == [{'symbol': 'AAAUSDT'}]
        return {'entry_threshold': .55, 'status': 'validated'}

    monkeypatch.setattr(strategy_comparison, 'select_entry_policy', select_policy)
    returns = iter([.05, -.10, .01, .90])
    monkeypatch.setattr(strategy_comparison, 'replay', lambda *args: {
        'net_return': next(returns), 'closed_trades': 25, 'halted': False, 'max_drawdown': -.02})
    result = strategy_comparison.compare({'AAAUSDT': frame}, [{'symbol': 'AAAUSDT'}], load_config())
    assert result['selected_on_validation'] == 'model'
    assert not result['test_supports_selected']


def test_comparison_evaluates_thresholds_with_executable_market_minimums(monkeypatch):
    frame = candles(1200)
    price = 100 + np.sin(np.arange(len(frame)) / 8)
    frame['open'], frame['close'] = price, price
    frame['high'], frame['low'] = price + 1, price - 1
    market = {'symbol': 'AAAUSDT', 'step_size': '.001', 'min_qty': '.001',
              'max_qty': '100000', 'min_notional': 10.}

    class Model:
        def fit(self, *args, **kwargs):
            pass

        def predict_proba(self, frame):
            return np.tile([.2, .8], (len(frame), 1))

    monkeypatch.setattr(strategy_comparison, 'XGBClassifier', lambda **kwargs: Model())
    result = strategy_comparison.compare({'AAAUSDT': frame}, [market], load_config())
    assert result['selected_on_validation'] is None
    assert result['threshold_policy']['status'] == 'no_validated_threshold'
    candidate = result['threshold_policy']['candidates'][0]
    assert candidate['evaluation_policy'] == 'shared_wallet_v1'
    assert candidate['trade_count'] == 0
    assert candidate['blocked_entries']


def test_comparison_publication_only_writes_research_diagnostics(monkeypatch):
    import base64
    import json

    calls = []

    class Store:
        branch = 'cryptoai-models'

        def __init__(self, branch):
            assert branch == self.branch

        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))

            class Response:
                status_code = 404 if method == 'GET' else 201

            return Response()

    monkeypatch.setattr(strategy_comparison, 'GitHubDeliveryState', Store)
    strategy_comparison.publish_result({'selected_on_validation': None})
    assert calls[1][1] == '/contents/research/strategy-comparison.json'
    payload = json.loads(base64.b64decode(calls[1][2]['json']['content']))
    assert payload == {'selected_on_validation': None}
