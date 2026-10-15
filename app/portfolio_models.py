import argparse
import base64
import io
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from app.core import FEATURES, INTERVAL, INTERVAL_MS, MODEL_DIR, label_data, make_features
from app.market import candle_frame, candle_windows, public_data, universe
from app.portfolio import load_config
from app.portfolio_evaluation import EVALUATION_POLICY, evaluation_passed, shared_wallet_results
from app.signals import GitHubDeliveryState

FILES = ('portfolio.joblib', 'portfolio-runtime.txt')
EXECUTION_POLICY = 'scheduled_quotes_v1'


def pooled_splits(frames, horizon):
    common = sorted(set.intersection(*(set(frame.open_time) for frame in frames.values())))
    if len(common) < 1000:
        raise ValueError('Insufficient shared history for chronological training.')
    a, b = common[int(len(common) * .6)], common[int(len(common) * .8)]
    parts = [[], [], []]
    for symbol, raw in frames.items():
        frame = label_data(make_features(raw), horizon, .004)
        frame['symbol'] = symbol
        future = frame.open_time + horizon * INTERVAL_MS
        parts[0].append(frame[future < a])
        parts[1].append(frame[(frame.open_time >= a) & (future < b)])
        parts[2].append(frame[frame.open_time >= b])
    return tuple(pd.concat(part, ignore_index=True) for part in parts)


def scheduled_trades(frame, scores, threshold, config):
    trades = []
    prices = frame.open.to_numpy(dtype=float)
    atrs = frame.atr.to_numpy(dtype=float)
    times = frame.open_time.to_numpy(dtype=int)
    fee, slip = config['fee_bps_per_side'] / 10000, config['slippage_bps_per_side'] / 10000
    spread = config.get('backtest_spread_bps', 2) / 20000
    horizon, next_allowed = config['horizon_bars'], 0
    for i, score in enumerate(scores):
        end = i + 1 + horizon
        if i < next_allowed or score < threshold or end >= len(frame):
            continue
        entry = prices[i + 1] * (1 + spread) * (1 + slip)
        atr = atrs[i]
        stop, target = entry - 1.5 * atr, entry + 3 * atr
        if not np.isfinite(atr) or atr <= 0 or stop <= 0:
            continue
        for j in range(i + 2, end + 1):
            bid = prices[j] * (1 - spread)
            reason = ('SL' if bid <= stop else 'TP' if bid >= target else 'TIME' if j == end
                      else 'MODEL' if scores[j - 1] < config['exit_threshold'] else None)
            if reason:
                exit_price = bid * (1 - slip)
                trades.append({'entry_time': int(times[i + 1]), 'exit_time': int(times[j]),
                               'result': reason, 'net_return': exit_price * (1 - fee) / (entry * (1 + fee)) - 1})
                next_allowed = j
                break
    return trades


def threshold_results(frame, scores, threshold, config, markets):
    return shared_wallet_results(frame, scores, threshold, config, markets)


def select_entry_policy(validation, scores, config, markets, minimum_trades=20):
    scores = np.asarray(scores, dtype=float)
    if (scores.shape != (len(validation),) or not len(scores)
            or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('Invalid validation scores.')
    ceiling, floor = config['entry_threshold'], config['exit_threshold']
    candidates = sorted({ceiling, *[float(value) for value in np.quantile(
        scores, [.5, .6, .7, .8, .9, .95, .99, .995, .999]) if floor < value <= ceiling]})
    evaluations = [{'threshold': threshold, **threshold_results(validation, scores, threshold, config, markets)}
                   for threshold in candidates]
    eligible = [result for result in evaluations if evaluation_passed(result, minimum_trades)]
    selected = max(eligible, key=lambda result: (result['net_return'], result['max_drawdown'], result['threshold'])) if eligible else None
    return {'entry_threshold': selected['threshold'] if selected else ceiling,
            'status': 'validated' if selected else 'no_validated_threshold',
            'minimum_trades': minimum_trades, 'validation': selected,
            'candidates': evaluations}


def split_metadata(train_set, validation, test, horizon):
    return {name: {'feature_start_time': int(part.open_time.min()),
                   'feature_end_time': int(part.open_time.max()),
                   'label_available_through_time': int(part.open_time.max()) + (horizon + 1) * INTERVAL_MS}
            for name, part in (('train', train_set), ('validation', validation), ('test', test))}


def model_config(bundle, config):
    resolved = dict(config)
    threshold = bundle.get('entry_threshold', config['entry_threshold'])
    if (not isinstance(threshold, (int, float)) or not np.isfinite(threshold)
            or not config['exit_threshold'] < threshold <= config['entry_threshold']):
        raise ValueError('Invalid model entry threshold; retrain after changing thresholds.')
    resolved['entry_threshold'] = threshold
    return resolved


def train():
    config = load_config()
    ranked = universe(config['universe_size'] + 25)
    markets = ranked[:config['universe_size']]
    now = int(public_data('/time')['serverTime'])
    frames, errors = candle_windows([item['symbol'] for item in markets], config['training_candles'], now)
    failures = {symbol: error for symbol, error in errors.items() if not error.startswith('Insufficient or stale candles')}
    if failures:
        raise ValueError(f'Incomplete training history: {json.dumps(failures)}')
    markets = [market for market in markets if market['symbol'] in frames]
    for candidate in ranked[config['universe_size']:]:
        if len(markets) == config['universe_size']:
            break
        try:
            frames[candidate['symbol']] = candle_frame(candidate['symbol'], config['training_candles'], now)
            markets.append(candidate)
        except ValueError:
            continue
    if len(markets) != config['universe_size']:
        raise ValueError('Not enough liquid markets with complete training history.')
    train_set, validation, test = pooled_splits(frames, config['horizon_bars'])
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.04,
                          subsample=.8, colsample_bytree=.9, random_state=42,
                          n_jobs=2, eval_metric='logloss')
    model.fit(train_set[FEATURES], train_set.target.astype(int),
              eval_set=[(validation[FEATURES], validation.target.astype(int))], verbose=False)
    policy = select_entry_policy(validation, model.predict_proba(validation[FEATURES])[:, 1], config, markets)
    scores = model.predict_proba(test[FEATURES])[:, 1]
    windows = split_metadata(train_set, validation, test, config['horizon_bars'])
    metrics = {'execution_policy': EXECUTION_POLICY, 'evaluation_policy': EVALUATION_POLICY,
               'data_windows': windows,
               'train_rows': len(train_set), 'validation_rows': len(validation),
               'test_rows': len(test), 'test_brier': float(brier_score_loss(test.target, scores)),
               'test_auc': float(roc_auc_score(test.target, scores)) if test.target.nunique() == 2 else None,
               'configured_entry_threshold': config['entry_threshold'],
               'entry_threshold': policy['entry_threshold'],
               'threshold_selection': policy,
               'qualifying_test_signals': int((scores >= policy['entry_threshold']).sum()),
               'test_score_max': float(scores.max()),
               'test_trade_results': threshold_results(test, scores, policy['entry_threshold'], config, markets)}
    bundle = {'schema': 1, 'model': model, 'features': FEATURES, 'interval': INTERVAL,
              'markets': markets, 'horizon': config['horizon_bars'], 'metrics': metrics,
              'entry_threshold': policy['entry_threshold'],
              'training_feature_end_time': windows['train']['feature_end_time'],
              'training_label_available_through_time': windows['train']['label_available_through_time'],
              'data_end_time': max(int(frame.open_time.iloc[-1]) for frame in frames.values()),
              'trained_at': datetime.now(timezone.utc).isoformat()}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / FILES[0])
    print(json.dumps({'symbols': [item['symbol'] for item in markets], **metrics}, indent=2))


def unpack(content):
    if len(content) > 500_000:
        raise ValueError('Portfolio bundle is too large.')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if sorted(archive.namelist()) != sorted(FILES) or sum(item.file_size for item in archive.infolist()) > 2_000_000:
            raise ValueError('Invalid portfolio bundle files.')
        payloads = {name: archive.read(name) for name in FILES}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (MODEL_DIR / name).write_bytes(payload)


def registry(command):
    store = GitHubDeliveryState(branch='cryptoai-models')
    path = '/contents/portfolio-bundle.zip'
    existing = store.request('GET', path, params={'ref': store.branch})
    if command == 'download':
        if existing.status_code != 200:
            raise RuntimeError('Run Train 50-market portfolio first.')
        unpack(base64.b64decode(existing.json()['content']))
        return
    if existing.status_code not in (200, 404):
        raise RuntimeError('Cannot read portfolio registry.')
    (MODEL_DIR / FILES[1]).write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(MODEL_DIR / name, name)
    content = output.getvalue()
    if len(content) > 500_000:
        raise ValueError('Portfolio bundle exceeds 500 KB.')
    body = {'branch': store.branch, 'message': 'Publish complete 50-market model',
            'content': base64.b64encode(content).decode()}
    if existing.status_code == 200:
        body['sha'] = existing.json()['sha']
    if store.request('PUT', path, json=body).status_code not in (200, 201):
        raise RuntimeError('Portfolio publication failed.')


def entry_policy_approved(bundle):
    metrics = bundle.get('metrics', {})
    selection = metrics.get('threshold_selection', {})
    return (metrics.get('execution_policy') == EXECUTION_POLICY
            and metrics.get('evaluation_policy') == EVALUATION_POLICY
            and selection.get('status') == 'validated'
            and evaluation_passed(selection.get('validation'))
            and evaluation_passed(metrics.get('test_trade_results')))


def paper_entries_allowed(bundle, config):
    if config.get('mode') != 'paper':
        return False
    if entry_policy_approved(bundle):
        return True
    if config.get('paper_entry_policy', 'validated') != 'after_training':
        return False
    metrics = bundle.get('metrics', {})
    return (bundle.get('schema') == 1 and bundle.get('features') == FEATURES
            and bundle.get('interval') == INTERVAL and bundle.get('horizon') == config['horizon_bars']
            and isinstance(bundle.get('trained_at'), str) and bool(bundle['trained_at'])
            and isinstance(bundle.get('data_end_time'), int) and bundle['data_end_time'] > 0
            and metrics.get('execution_policy') == EXECUTION_POLICY
            and all(isinstance(metrics.get(key), int) and metrics[key] > 0
                    for key in ('train_rows', 'validation_rows', 'test_rows')))


def predict(symbol, frame, bundle, now, config):
    config = model_config(bundle, config)
    if bundle.get('schema') != 1 or bundle.get('features') != FEATURES or bundle.get('interval') != INTERVAL:
        raise ValueError('Invalid pooled model schema.')
    candle = int(frame.open_time.iloc[-1])
    if candle != now // INTERVAL_MS * INTERVAL_MS - INTERVAL_MS:
        raise ValueError('Stale candle.')
    if bundle['data_end_time'] >= candle:
        return {'symbol': symbol, 'action': 'WAITING'}
    if bundle['horizon'] != config['horizon_bars']:
        raise ValueError('Retrain after changing the horizon.')
    latest = make_features(frame).iloc[[-1]]
    if not np.isfinite(latest[FEATURES].to_numpy()).all():
        raise ValueError('Invalid latest features.')
    score = float(bundle['model'].predict_proba(latest[FEATURES])[0, 1])
    atr = float(latest.atr.iloc[0])
    if not np.isfinite(score) or not 0 <= score <= 1 or not np.isfinite(atr) or atr <= 0:
        raise ValueError('Invalid model output.')
    entry_allowed = paper_entries_allowed(bundle, config)
    return {'symbol': symbol, 'action': 'BUY' if entry_allowed and score >= config['entry_threshold'] else 'HOLD',
            'entry_allowed': entry_allowed,
            'model_score': score, 'atr': atr, 'candle_time': candle, 'horizon_bars': bundle['horizon']}


def ensure_horizon():
    bundle = joblib.load(MODEL_DIR / FILES[0])
    if (bundle.get('horizon') == load_config()['horizon_bars']
            and bundle.get('metrics', {}).get('execution_policy') == EXECUTION_POLICY):
        return
    train()
    registry('upload')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'upload', 'download', 'ensure-horizon'])
    args = parser.parse_args()
    try:
        if args.command == 'ensure-horizon':
            ensure_horizon()
        elif args.command == 'train':
            train()
        else:
            registry(args.command)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')
