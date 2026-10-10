import numpy as np
import pandas as pd
from app.core import make_features, label_data, split_time, simulate, INTERVAL_MS


def synthetic(n=1000):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({'open_time': np.arange(n)*INTERVAL_MS,
                         'open': opening, 'high': np.maximum(opening, close)*1.002,
                         'low': np.minimum(opening, close)*0.998,
                         'close': close, 'volume': rng.uniform(10, 100, n)})


def test_features_and_purged_splits():
    d = label_data(make_features(synthetic()), horizon=12)
    a, b, c = split_time(d, 12)
    assert len(a) > 0 and len(b) > 0 and len(c) > 0
    assert a.index.max() + 12 < b.index.min()
    assert b.index.max() + 12 < c.index.min()
    assert d.iloc[-1]['open_time'] <= (1000-13)*INTERVAL_MS


def test_simulator_never_overlaps():
    d = make_features(synthetic())
    trades = simulate(d, np.ones(len(d)), horizon=12)
    for a, b in zip(trades, trades[1:]):
        assert a['exit_time'] < b['signal_time']
