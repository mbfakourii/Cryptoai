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
from app.signals import GitHubDeliveryState

FILES = ('portfolio.joblib', 'portfolio-runtime.txt')


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
    scores = model.predict_proba(test[FEATURES])[:, 1]
    metrics = {'train_rows': len(train_set), 'validation_rows': len(validation),
               'test_rows': len(test), 'test_brier': float(brier_score_loss(test.target, scores)),
               'test_auc': float(roc_auc_score(test.target, scores)) if test.target.nunique() == 2 else None,
               'qualifying_test_signals': int((scores >= config['entry_threshold']).sum())}
    bundle = {'schema': 1, 'model': model, 'features': FEATURES, 'interval': INTERVAL,
              'markets': markets, 'horizon': config['horizon_bars'], 'metrics': metrics,
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


def predict(symbol, frame, bundle, now, config):
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
    return {'symbol': symbol, 'action': 'BUY' if score >= config['entry_threshold'] else 'HOLD',
            'model_score': score, 'atr': atr, 'candle_time': candle, 'horizon_bars': bundle['horizon']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'upload', 'download'])
    args = parser.parse_args()
    try:
        train() if args.command == 'train' else registry(args.command)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')
