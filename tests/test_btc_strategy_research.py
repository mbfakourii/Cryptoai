import numpy as np
import pandas as pd
import pytest

from app.btc_strategy_research import execution_returns, forecast_error, policy_scores, return_features, select_candidate
from app.core import INTERVAL_MS
from app.portfolio import load_config, new_wallet, paper_cycle


@pytest.mark.parametrize('opens', [[100.] * 8, [100., 100., 95., 100., 100., 100., 100., 100.],
                                   [100., 100., 110., 100., 100., 100., 100., 100.]])
def test_training_return_matches_scheduled_execution_after_all_costs(opens):
    config = load_config()
    frame = pd.DataFrame({'open': opens, 'atr': [1.] * len(opens)})
    expected = execution_returns(frame, config).iloc[0]
    market = {'symbol': 'BTCUSDT', 'step_size': '.001', 'min_qty': '.001',
              'max_qty': '100000', 'min_notional': 5.}
    wallet = new_wallet(config)
    entry_cost = None
    for step in range(1, 6):
        price = opens[step]
        quotes = {'BTCUSDT': {'bid': price * .9999, 'ask': price * 1.0001}}
        signal = {'symbol': 'BTCUSDT', 'action': 'BUY' if step == 1 else 'HOLD',
                  'model_score': .8 if step == 1 else .5, 'atr': 1.,
                  'candle_time': (step - 1) * INTERVAL_MS, 'horizon_bars': 4}
        wallet, events = paper_cycle(wallet, {'BTCUSDT': signal}, quotes,
                                      {'BTCUSDT': market}, {}, step * INTERVAL_MS, config)
        if step == 1:
            entry_cost = wallet['positions']['BTCUSDT']['cost']
        sell = next((item for item in events if item['side'] == 'SELL'), None)
        if sell:
            assert sell['pnl'] / entry_cost * 10000 == pytest.approx(expected)
            return
    pytest.fail('Expected an exit within the configured horizon.')


def test_missing_future_returns_are_not_training_targets():
    frame = pd.DataFrame({'open': [100.] * 10, 'atr': [1.] * 10})
    returns = execution_returns(frame, load_config())
    assert returns.iloc[-5:].isna().all()
    assert returns.iloc[:-5].notna().all()


def test_features_do_not_change_when_future_prices_arrive():
    rng = np.random.default_rng(42)
    close = 100 * np.exp(rng.normal(0, .01, 200).cumsum())
    raw = pd.DataFrame({'open_time': np.arange(200) * INTERVAL_MS,
                        'open': close, 'high': close * 1.01, 'low': close * .99,
                        'close': close, 'volume': rng.uniform(100, 1000, 200)})
    pd.testing.assert_frame_equal(return_features(raw.iloc[:150]), return_features(raw).iloc[:150])


def test_negative_expected_returns_do_not_create_buys():
    frame = pd.DataFrame({'ema_gap': [1.] * 4})
    scores = policy_scores(np.array([-20., -5., 5., 15.]), frame,
                            {'kind': 'net_return', 'minimum_net_bps': 10, 'model_exit': True})
    np.testing.assert_equal(scores, [.2, .5, .5, .8])


def test_two_lucky_wins_are_not_a_validated_strategy():
    candidates = [{'validation': {'closed_trades': 2, 'net_return': .01,
                                  'max_drawdown': -.001, 'halted': False}}]
    assert select_candidate(candidates) is None


def test_forecast_quality_compares_against_training_mean_without_immature_targets():
    error = forecast_error([10., -10., np.nan], [0., 0., 100.], 0.)
    assert error == {'rows': 2, 'model_rmse_bps': 10., 'training_mean_rmse_bps': 10.}


def test_forecast_quality_requires_mature_targets():
    with pytest.raises(ValueError, match='No mature forecast targets'):
        forecast_error([np.nan], [10.], 0.)
