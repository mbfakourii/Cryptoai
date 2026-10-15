import numpy as np
import pandas as pd
import pytest

from app.btc_refresh_research import refreshed_predictions, same_result, summarize, training_rows
from app.btc_strategy_research import RETURN_FEATURES, execution_returns, return_features
from app.btc_walk_forward import DAY_MS
from app.core import INTERVAL_MS
from app.portfolio import load_config


def labelled_frame(rows=500):
    frame = pd.DataFrame({name: np.arange(rows, dtype=float) for name in RETURN_FEATURES})
    frame['open_time'] = np.arange(rows) * INTERVAL_MS
    frame['net_return_bps'] = np.arange(rows, dtype=float)
    return frame


class MeanModel:
    def fit(self, features, target):
        self.mean = float(target.mean())

    def predict(self, features):
        return np.full(len(features), self.mean)


def test_refresh_uses_only_mature_labels_and_the_configured_rolling_window():
    frame = labelled_frame(600)
    start, end = 400 * INTERVAL_MS, 419 * INTERVAL_MS
    values, fits = refreshed_predictions(frame, start, end, 4, 8, training_days=1, model_factory=MeanModel)
    assert len(fits) == 3
    assert [item['prediction_rows'] for item in fits] == [8, 8, 3]
    for fit in fits:
        assert fit['training_feature_start'] == fit['prediction_start'] - DAY_MS
        assert fit['training_feature_end'] == fit['prediction_start'] - 6 * INTERVAL_MS
        assert fit['training_labels_available_through'] == fit['prediction_start'] - INTERVAL_MS
        assert fit['training_rows'] == 91
        part = (frame.open_time >= fit['prediction_start']) & (frame.open_time < fit['prediction_end_exclusive'])
        expected = training_rows(frame, fit['training_feature_start'], fit['prediction_start'], 4).net_return_bps.mean()
        assert (values[part] == expected).all()
    assert np.isnan(values[:400]).all()
    assert np.isnan(values[419:]).all()


def test_unavailable_future_labels_cannot_affect_earlier_predictions():
    frame = labelled_frame()
    changed = frame.copy()
    start, end = 300 * INTERVAL_MS, 324 * INTERVAL_MS
    changed.loc[changed.open_time >= start - 5 * INTERVAL_MS, 'net_return_bps'] = 1e12
    original, _ = refreshed_predictions(frame, start, end, 4, 8, model_factory=MeanModel)
    poisoned, _ = refreshed_predictions(changed, start, end, 4, 8, model_factory=MeanModel)
    np.testing.assert_array_equal(original[300:308], poisoned[300:308])
    assert not np.array_equal(original[308:316], poisoned[308:316])


def test_future_price_changes_do_not_change_previous_refresh_predictions():
    rows = 600
    prices = 100 + np.sin(np.arange(rows) / 11)
    raw = pd.DataFrame({'open_time': np.arange(rows) * INTERVAL_MS,
                        'open': prices, 'high': prices + 1, 'low': prices - 1,
                        'close': prices, 'volume': np.full(rows, 100.)})
    poisoned = raw.copy()
    poisoned.loc[330:, ['open', 'high', 'low', 'close']] *= 2
    results = []
    for candles in (raw, poisoned):
        featured = return_features(candles)
        featured['net_return_bps'] = execution_returns(featured, load_config())
        values, _ = refreshed_predictions(featured, 300 * INTERVAL_MS, 360 * INTERVAL_MS,
                                           4, 10, model_factory=MeanModel)
        results.append(values)
    np.testing.assert_array_equal(results[0][300:330], results[1][300:330])
    assert not np.array_equal(results[0][340:350], results[1][340:350])


def test_weekly_cadence_only_refits_at_its_boundary():
    frame = labelled_frame(2000)
    _, fits = refreshed_predictions(frame, 500 * INTERVAL_MS, 1900 * INTERVAL_MS,
                                     4, 7 * 96, model_factory=MeanModel)
    assert [item['prediction_start'] // INTERVAL_MS for item in fits] == [500, 1172, 1844]
    assert [item['prediction_rows'] for item in fits] == [672, 672, 56]


@pytest.mark.parametrize('cadence,days,horizon', [(0, 180, 4), (1, 0, 4), (1, 180, 0)])
def test_invalid_refresh_settings_are_rejected(cadence, days, horizon):
    with pytest.raises(ValueError, match='must be positive'):
        refreshed_predictions(labelled_frame(), 300 * INTERVAL_MS, 320 * INTERVAL_MS,
                              horizon, cadence, days, MeanModel)


def test_refresh_does_not_fit_when_no_mature_labels_exist():
    with pytest.raises(ValueError, match='No mature training labels'):
        refreshed_predictions(labelled_frame(), 4 * INTERVAL_MS, 20 * INTERVAL_MS,
                              4, 8, model_factory=MeanModel)


def test_invalid_predictions_are_not_silently_reported_as_cash_returns():
    class InvalidModel(MeanModel):
        def predict(self, features):
            return np.full(len(features), np.nan)

    with pytest.raises(ValueError, match='finite prediction'):
        refreshed_predictions(labelled_frame(), 300 * INTERVAL_MS, 320 * INTERVAL_MS,
                              4, 8, model_factory=InvalidModel)


def test_frozen_reproduction_checks_counts_and_returns():
    expected = {'closed_trades': 3, 'net_return': .001, 'halted': False}
    assert same_result(expected, expected)
    assert not same_result({**expected, 'closed_trades': 2}, expected)
    assert not same_result({**expected, 'net_return': .002}, expected)
    assert not same_result({**expected, 'halted': True}, expected)


def test_summary_compounds_independent_folds_without_claiming_approval():
    def strategy(result):
        stats = {'net_return': result, 'closed_trades': 2, 'wins': 1, 'fees_paid': .01,
                 'max_drawdown': -.03, 'halted': False}
        return {'test': stats, 'test_higher_costs': {**stats, 'net_return': result - .01},
                'fits': [{'fit_seconds': .1}]}

    folds = [{'strategies': {'frozen': strategy(-.01), 'daily_refresh': strategy(.02)}},
             {'strategies': {'frozen': strategy(.01), 'daily_refresh': strategy(-.03)}}]
    summary = summarize(folds)['daily_refresh']
    assert summary['normalized_compounded_return'] == pytest.approx(1.02 * .97 - 1)
    assert summary['higher_cost_compounded_return'] == pytest.approx(1.01 * .96 - 1)
    assert summary['paired_folds_better_than_frozen'] == 1
    assert summary['fit_count'] == 2
    assert summary['closed_trades'] == 4
    assert not any('approved' in key for key in summary)
