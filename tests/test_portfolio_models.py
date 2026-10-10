import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.core import INTERVAL_MS
from app.portfolio_models import pooled_splits, unpack


def test_pooled_training_purges_future_labels_across_all_markets():
    rng = np.random.default_rng(12)
    frames = {}
    for symbol in ('AAAUSDT', 'BBBUSDT'):
        close = 100 * np.exp(np.cumsum(rng.normal(0, .005, 1200)))
        frames[symbol] = pd.DataFrame({'open_time': np.arange(1200) * INTERVAL_MS,
                                      'open': close, 'close': close, 'high': close * 1.01,
                                      'low': close * .99, 'volume': rng.uniform(10, 100, 1200)})
    train, validation, test = pooled_splits(frames, 12)
    assert train.open_time.max() + 12 * INTERVAL_MS < validation.open_time.min()
    assert validation.open_time.max() + 12 * INTERVAL_MS < test.open_time.min()
    for part in (train, validation, test):
        assert set(part.symbol) == set(frames)


def test_bundle_rejects_unexpected_paths_before_extracting():
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        archive.writestr('../unexpected', b'x')
    with pytest.raises(ValueError):
        unpack(output.getvalue())
