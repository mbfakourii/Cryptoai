import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

from app.btc_strategy_research import RETURN_FEATURES, execution_returns, policy_scores, return_features
from app.btc_walk_forward import DAY_MS, replay_window, validate_history
from app.core import DATA, INTERVAL_MS

MODEL_PARAMETERS = {'n_estimators': 200, 'max_depth': 2, 'learning_rate': .03,
                    'min_child_weight': 50, 'reg_lambda': 10, 'subsample': .8,
                    'colsample_bytree': .9, 'random_state': 42, 'n_jobs': 2,
                    'objective': 'reg:squarederror'}


def return_model():
    return XGBRegressor(**MODEL_PARAMETERS)


def training_rows(featured, start, prediction_start, horizon):
    maturity = featured.open_time + (horizon + 1) * INTERVAL_MS
    train = featured[(featured.open_time >= start) & (maturity < prediction_start)]
    train = train.dropna(subset=RETURN_FEATURES + ['net_return_bps'])
    if train.empty:
        raise ValueError('No mature training labels before the prediction batch.')
    return train


def predict_values(model, features):
    values = np.asarray(model.predict(features), dtype=float)
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError('Expected one finite prediction per candle.')
    return values


def refreshed_predictions(featured, start, end, horizon, cadence_bars,
                          training_days=180, model_factory=return_model):
    if start >= end or cadence_bars <= 0 or training_days <= 0 or horizon <= 0:
        raise ValueError('Prediction range, cadence, training window and horizon must be positive.')
    values = np.full(len(featured), np.nan)
    fits = []
    for batch_start in range(start, end, cadence_bars * INTERVAL_MS):
        batch_end = min(batch_start + cadence_bars * INTERVAL_MS, end)
        train = training_rows(featured, batch_start - training_days * DAY_MS, batch_start, horizon)
        indices = np.flatnonzero((featured.open_time >= batch_start) & (featured.open_time < batch_end))
        if not len(indices):
            raise ValueError('Prediction batch contains no candles.')
        model = model_factory()
        started = time.perf_counter()
        model.fit(train[RETURN_FEATURES], train.net_return_bps)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        values[indices] = predict_values(model, featured.iloc[indices][RETURN_FEATURES])
        fits.append({'prediction_start': batch_start, 'prediction_end_exclusive': batch_end,
                     'training_feature_start': int(train.open_time.min()),
                     'training_feature_end': int(train.open_time.max()),
                     'training_labels_available_through': int(train.open_time.max()) + (horizon + 1) * INTERVAL_MS,
                     'training_rows': len(train), 'prediction_rows': len(indices),
                     'fit_seconds': fit_seconds, 'prediction_seconds': time.perf_counter() - started})
    return values, fits


def same_result(actual, expected):
    return all(key in actual and (actual[key] == value if isinstance(value, (bool, int))
                                  else np.isclose(actual[key], value, rtol=0, atol=1e-10))
               for key, value in expected.items())


def summarize(folds):
    summary = {}
    for name in folds[0]['strategies']:
        results = [fold['strategies'][name] for fold in folds]
        tests = [item['test'] for item in results]
        summary[name] = {
            'test_folds': len(tests),
            'profitable_test_folds': sum(item['net_return'] > 0 for item in tests),
            'closed_trades': sum(item['closed_trades'] for item in tests),
            'wins': sum(item['wins'] for item in tests),
            'fees_paid_across_independent_wallets': sum(item['fees_paid'] for item in tests),
            'worst_test_drawdown': min(item['max_drawdown'] for item in tests),
            'halted_folds': sum(item['halted'] for item in tests),
            'mean_test_return': float(np.mean([item['net_return'] for item in tests])),
            'median_test_return': float(np.median([item['net_return'] for item in tests])),
            'normalized_compounded_return': float(np.prod([1 + item['net_return'] for item in tests]) - 1),
            'higher_cost_compounded_return': float(np.prod([1 + item['test_higher_costs']['net_return']
                                                           for item in results]) - 1),
            'fit_count': sum(len(item['fits']) for item in results),
            'fit_seconds': sum(fit['fit_seconds'] for item in results for fit in item['fits']),
            'paired_folds_better_than_frozen': sum(
                fold['strategies'][name]['test']['net_return'] > fold['strategies']['frozen']['test']['net_return']
                for fold in folds),
        }
    return summary


def compare(raw, source, progress=None):
    start = int(pd.Timestamp(source['source']['start_inclusive'], tz='UTC').timestamp() * 1000)
    end = int(pd.Timestamp(source['source']['end_exclusive'], tz='UTC').timestamp() * 1000)
    validate_history(raw, start, end)
    if source['rows'] != len(raw) or source['symbol'] != 'BTCUSDT' or not source['folds']:
        raise ValueError('Expected the complete BTC walk-forward source report.')
    config, market = source['config'], source['market_rules']
    horizon = config['horizon_bars']
    featured = return_features(raw)
    featured['net_return_bps'] = execution_returns(featured, config)
    folds = []
    for old in source['folds']:
        bounds = {key: int(pd.Timestamp(value).timestamp() * 1000) for key, value in old['bounds_utc'].items()}
        previous = old['strategies']['net_return']
        policy = previous['policy'].copy()
        train = training_rows(featured, bounds['train_start'], bounds['validation_start'], horizon)
        model = return_model()
        started = time.perf_counter()
        model.fit(train[RETURN_FEATURES], train.net_return_bps)
        fit_seconds = time.perf_counter() - started
        test_indices = np.flatnonzero((featured.open_time >= bounds['test_start'])
                                     & (featured.open_time < bounds['test_end']))
        predictions = np.full(len(featured), np.nan)
        predictions[test_indices] = predict_values(model, featured.iloc[test_indices][RETURN_FEATURES])
        scores = policy_scores(predictions, featured, policy)
        normal = replay_window(featured, scores, market, bounds['test_start'], bounds['test_end'], policy, config)
        higher_costs = {**config, 'fee_bps_per_side': 15, 'slippage_bps_per_side': 4}
        stress = replay_window(featured, scores, market, bounds['test_start'], bounds['test_end'], policy, higher_costs)
        if not same_result(normal, previous['test']) or not same_result(stress, previous['test_higher_costs']):
            raise ValueError(f'Frozen reproduction differs from the source report in fold {old["number"]}.')
        frozen_fit = {'training_feature_start': int(train.open_time.min()),
                      'training_feature_end': int(train.open_time.max()),
                      'training_labels_available_through': int(train.open_time.max()) + (horizon + 1) * INTERVAL_MS,
                      'training_rows': len(train), 'fit_seconds': fit_seconds}
        fold = {'number': old['number'], 'bounds_utc': old['bounds_utc'], 'fixed_policy': policy,
                'source_validation_passed': previous['validation_passed'],
                'source_frozen_test': previous['test'], 'frozen_reproduction_matches': True,
                'strategies': {'frozen': {'test': normal, 'test_higher_costs': stress, 'fits': [frozen_fit]}}}
        for name, cadence in (('daily_refresh', 96), ('weekly_refresh', 7 * 96)):
            predictions, fits = refreshed_predictions(featured, bounds['test_start'], bounds['test_end'],
                                                       horizon, cadence, source['design']['training_days'])
            scores = policy_scores(predictions, featured, policy)
            fold['strategies'][name] = {
                'test': replay_window(featured, scores, market, bounds['test_start'], bounds['test_end'], policy, config),
                'test_higher_costs': replay_window(featured, scores, market, bounds['test_start'], bounds['test_end'],
                                                   policy, higher_costs), 'fits': fits}
        folds.append(fold)
        if progress:
            progress(fold)
    return {'symbol': source['symbol'], 'rows': len(raw), 'folds': folds,
            'summary': summarize(folds), 'config': config, 'market_rules': market,
            'design': {'training_days': source['design']['training_days'],
                       'refresh_cadence_bars': {'daily_refresh': 96, 'weekly_refresh': 672},
                       'prediction_horizon_bars': horizon,
                       'label_maturity_rule': 'feature_open_time + (horizon + 1) * interval < prediction_batch_start',
                       'policy_selection': 'Frozen per-fold net-return policy from the source validation window.',
                       'refreshed_validation_approval': False, 'approved_for_live_activation': False,
                       'regressor_parameters': MODEL_PARAMETERS.copy()},
            'limitations': [
                'Previously inspected historical dates; an exploratory paired comparison, not fresh forward confirmation.',
                'Refreshing is a new procedure and does not inherit the frozen model validation approval.',
                'BTC technical features only; not the deployed pooled model or historical news/macro context.',
                'One-hour net-return labels are unchanged even when the frozen policy holds for longer.',
                'Next-open scheduled quotes and historical use of current order rules approximate exchange execution.',
                'Independent wallets start each fold; normalized compounding is not one continuous wallet.',
                'Daily and weekly intervals are fixed hypotheses; no thresholds or cadence tuned on test returns.',
                'No model publication, wallet mutation, or live activation.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=Path, default=DATA / 'research' / 'btc-2024-10-01-2026-10-01.csv')
    parser.add_argument('--source-report', type=Path, default=Path('research/btc-walk-forward-2024-2026.json'))
    parser.add_argument('--output', type=Path, default=Path('research/btc-refresh-2024-2026.json'))
    args = parser.parse_args()
    source_bytes = args.source_report.read_bytes()
    source = json.loads(source_bytes)
    csv_hash = hashlib.sha256(args.history.read_bytes()).hexdigest()
    if csv_hash != source['source']['csv_sha256']:
        raise ValueError('Historical CSV does not match the source report hash.')

    def progress(fold):
        print(json.dumps({'fold': fold['number'], 'tests': {name: item['test']['net_return']
                          for name, item in fold['strategies'].items()}}), file=sys.stderr, flush=True)

    started = time.perf_counter()
    result = compare(pd.read_csv(args.history), source, progress)
    result['elapsed_seconds'] = time.perf_counter() - started
    result['source'] = {**source['source'], 'source_report': args.source_report.as_posix(),
                        'source_report_sha256': hashlib.sha256(source_bytes).hexdigest(),
                        'csv_sha256': csv_hash}
    result['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'xgboost': xgboost.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
