import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.core import INTERVAL_MS
from app.portfolio import load_config
from app.portfolio_models import model_config, pooled_splits, select_entry_policy, unpack
from app import portfolio_models


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


def validation_fixture(profitable=True):
    prices = 100 + np.arange(120) * (.5 if profitable else -.2)
    return pd.DataFrame({'symbol': ['AAAUSDT'] * 120,
                         'open_time': np.arange(120) * INTERVAL_MS,
                         'open': prices, 'close': prices,
                         'high': prices + .1, 'low': prices - .1, 'atr': .1})


def test_threshold_selection_can_find_validated_entries_below_the_fixed_cutoff():
    config = load_config()
    policy = select_entry_policy(validation_fixture(), np.full(120, .45), config)
    assert policy['status'] == 'validated'
    assert config['exit_threshold'] < policy['entry_threshold'] < config['entry_threshold']
    assert policy['validation']['trade_count'] >= policy['minimum_trades']
    assert policy['validation']['mean_net_return'] > 0


def test_threshold_selection_does_not_force_entries_on_losing_validation_trades():
    config = load_config()
    policy = select_entry_policy(validation_fixture(False), np.full(120, .45), config)
    assert policy['status'] == 'no_validated_threshold'
    assert policy['entry_threshold'] == config['entry_threshold']
    assert policy['validation'] is None


def test_threshold_selection_requires_enough_validation_trades():
    policy = select_entry_policy(validation_fixture().iloc[:4], np.full(4, .45), load_config())
    assert policy['status'] == 'no_validated_threshold'


@pytest.mark.parametrize('scores', [np.full(120, np.nan), np.full(120, 1.1), np.full(119, .45)])
def test_threshold_selection_rejects_invalid_validation_scores(scores):
    with pytest.raises(ValueError, match='validation scores'):
        select_entry_policy(validation_fixture(), scores, load_config())


def test_saved_threshold_is_used_without_changing_risk_limits():
    config = load_config()
    resolved = model_config({'entry_threshold': .45}, config)
    assert resolved['entry_threshold'] == .45
    assert config['entry_threshold'] == .65
    for key in config.keys() - {'entry_threshold'}:
        assert resolved[key] == config[key]


def test_legacy_models_keep_the_configured_threshold_until_retrained():
    config = load_config()
    assert model_config({}, config) == config


@pytest.mark.parametrize('threshold', [0, .4, .9, float('nan'), '0.45'])
def test_saved_threshold_cannot_bypass_configured_bounds(threshold):
    with pytest.raises(ValueError, match='entry threshold'):
        model_config({'entry_threshold': threshold}, load_config())


def test_configured_portfolio_horizon_is_one_hour():
    assert load_config()['horizon_bars'] * INTERVAL_MS == 3600000


def test_old_horizon_is_retrained_before_the_matching_model_is_published(monkeypatch):
    calls = []
    monkeypatch.setattr(portfolio_models.joblib, 'load', lambda path: {'horizon': 12})
    monkeypatch.setattr(portfolio_models, 'train', lambda: calls.append('train'))
    monkeypatch.setattr(portfolio_models, 'registry', calls.append)
    portfolio_models.ensure_horizon()
    assert calls == ['train', 'upload']


def test_matching_horizon_does_not_retrain_every_cycle(monkeypatch):
    monkeypatch.setattr(portfolio_models.joblib, 'load', lambda path: {'horizon': 4, 'metrics': {'execution_policy': portfolio_models.EXECUTION_POLICY}})
    monkeypatch.setattr(portfolio_models, 'train', lambda: pytest.fail('Unexpected retraining'))
    monkeypatch.setattr(portfolio_models, 'registry', lambda command: pytest.fail('Unexpected publication'))
    portfolio_models.ensure_horizon()


def test_failed_retraining_cannot_publish_a_mismatched_model(monkeypatch):
    monkeypatch.setattr(portfolio_models.joblib, 'load', lambda path: {'horizon': 12})

    def fail():
        raise RuntimeError('Unavailable candles')

    monkeypatch.setattr(portfolio_models, 'train', fail)
    monkeypatch.setattr(portfolio_models, 'registry', lambda command: pytest.fail('Unexpected publication'))
    with pytest.raises(RuntimeError, match='Unavailable'):
        portfolio_models.ensure_horizon()


@pytest.mark.parametrize('count,net_return,allowed', [(20, .001, True), (19, .01, False),
                                                     (50, -.001, False), (0, None, False)])
def test_entry_approval_requires_positive_unseen_test_returns(count, net_return, allowed):
    bundle = {'metrics': {'execution_policy': portfolio_models.EXECUTION_POLICY,
                           'threshold_selection': {'status': 'validated'},
                           'test_trade_results': {'trade_count': count, 'mean_net_return': net_return}}}
    assert portfolio_models.entry_policy_approved(bundle) is allowed


def test_old_intrabar_backtests_do_not_authorize_scheduled_quote_entries():
    bundle = {'metrics': {'threshold_selection': {'status': 'validated'},
                           'test_trade_results': {'trade_count': 100, 'mean_net_return': .01}}}
    assert not portfolio_models.entry_policy_approved(bundle)


def test_matching_horizon_with_old_execution_rules_is_retrained(monkeypatch):
    calls = []
    monkeypatch.setattr(portfolio_models.joblib, 'load', lambda path: {'horizon': 4})
    monkeypatch.setattr(portfolio_models, 'train', lambda: calls.append('train'))
    monkeypatch.setattr(portfolio_models, 'registry', calls.append)
    portfolio_models.ensure_horizon()
    assert calls == ['train', 'upload']


def trained_losing_bundle():
    return {'schema': 1, 'features': portfolio_models.FEATURES, 'interval': portfolio_models.INTERVAL,
            'horizon': 4, 'trained_at': '2026-10-01T00:00:00Z', 'data_end_time': 100 * INTERVAL_MS,
            'metrics': {'execution_policy': portfolio_models.EXECUTION_POLICY,
                        'train_rows': 1000, 'validation_rows': 300, 'test_rows': 300,
                        'threshold_selection': {'status': 'no_validated_threshold'},
                        'test_trade_results': {'trade_count': 30, 'mean_net_return': -.01}}}


def test_explicit_paper_experiment_allows_trained_model_without_claiming_profit_approval():
    bundle = trained_losing_bundle()
    config = {**load_config(), 'paper_entry_policy': 'after_training'}
    assert portfolio_models.paper_entries_allowed(bundle, config)
    assert not portfolio_models.entry_policy_approved(bundle)


@pytest.mark.parametrize('policy,mode', [('validated', 'paper'), ('after_training', 'live'),
                                       ('after_training', None), ('unknown', 'paper')])
def test_experimental_training_permission_is_explicit_and_paper_only(policy, mode):
    config = {**load_config(), 'paper_entry_policy': policy, 'mode': mode}
    assert not portfolio_models.paper_entries_allowed(trained_losing_bundle(), config)


@pytest.mark.parametrize('missing', ['schema', 'features', 'interval', 'horizon', 'trained_at', 'data_end_time', 'metrics'])
def test_incomplete_training_cannot_enable_experimental_paper_entries(missing):
    bundle = trained_losing_bundle()
    del bundle[missing]
    assert not portfolio_models.paper_entries_allowed(bundle, {**load_config(), 'paper_entry_policy': 'after_training'})


def test_training_without_completed_evaluation_is_not_ready_for_paper_experiment():
    bundle = trained_losing_bundle()
    bundle['metrics']['test_rows'] = 0
    assert not portfolio_models.paper_entries_allowed(bundle, {**load_config(), 'paper_entry_policy': 'after_training'})
