import copy

import numpy as np
import pandas as pd

from app import portfolio_runner as runner
from app.core import INTERVAL_MS
from app.market_context import CONTEXT_DIMENSIONS
from app.portfolio import load_config, new_wallet


def test_runner_persists_learning_with_wallet_and_resumes_on_next_mature_candle(monkeypatch):
    config = load_config()
    market = {'symbol': 'AAAUSDT', 'step_size': '.01', 'min_qty': '.01',
              'max_qty': '10000', 'min_notional': 5.}
    markets = [{**market, 'symbol': f'MARKET{i}USDT'} for i in range(49)] + [market]
    bundle = {'markets': markets, 'metrics': {}, 'trained_at': '2026-10-01T00:00:00Z'}
    state = new_wallet(config)
    writes = []
    now = [100 * INTERVAL_MS + 30000]
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .002, 130)))
    history = pd.DataFrame({'open_time': np.arange(130) * INTERVAL_MS, 'open': close,
                            'close': close, 'high': close * 1.01, 'low': close * .99,
                            'volume': rng.uniform(10, 100, 130)})

    class Store:
        def read(self, config):
            return copy.deepcopy(writes[-1] if writes else state), 'sha'

        def write(self, state, sha):
            writes.append(copy.deepcopy(state))
            return 'next-sha'

    def candles(*args):
        return {'AAAUSDT': history[history.open_time + INTERVAL_MS <= now[0]].copy()}, {}

    def predict(symbol, frame, bundle, now, config):
        return {'symbol': symbol, 'action': 'HOLD', 'model_score': .3,
                'candle_time': int(frame.open_time.iloc[-1]), 'atr': 1., 'horizon_bars': 4}

    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test-token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test-chat')
    monkeypatch.setattr(runner.joblib, 'load', lambda path: bundle)
    monkeypatch.setattr(runner, 'WalletStore', Store)
    monkeypatch.setattr(runner, 'public_data', lambda path: {'serverTime': now[0]})
    monkeypatch.setattr(runner, 'candle_windows', candles)
    monkeypatch.setattr(runner, 'predict', predict)
    monkeypatch.setattr(runner, 'book_quotes', lambda symbols: {symbol: {'bid': 100., 'ask': 100.01} for symbol in symbols})
    monkeypatch.setattr(runner, 'collect_context', lambda now: {
        'ready': True, 'vector': [0.] * CONTEXT_DIMENSIONS, 'errors': {},
        'observed_at': now, 'macro': {}, 'news': [], 'news_sources': ['test']})
    runner.run()
    assert writes[-1]['cash'] == 20.
    assert writes[-1]['learning_summary']['pending_samples'] == 1
    runner.run()
    assert len(writes) == 1
    now[0] += 4 * INTERVAL_MS
    runner.run()
    assert len(writes) == 2
    assert writes[-1]['learning_summary']['market_samples'] == 1
    assert writes[-1]['learning_summary']['pending_samples'] == 1
    assert writes[-1]['market_context']['ready']
