import argparse
import base64
import json
import os
from datetime import datetime, timezone

import numpy as np
from xgboost import XGBClassifier

from app.core import FEATURES, INTERVAL_MS, make_features
from app.market import candle_windows, public_data, universe
from app.portfolio import load_config, new_wallet, paper_cycle
from app.portfolio_models import pooled_splits, select_entry_policy
from app.signals import GitHubDeliveryState


def trend_scores(raw):
    close = raw.close
    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=50, adjust=False).mean()
    previous_high = raw.high.shift(1).rolling(20).max()
    volume_ratio = raw.volume / raw.volume.rolling(20).mean()
    entry = (close > previous_high) & (fast > slow) & (volume_ratio >= 1)
    return np.where(entry, .8, np.where(close < fast, .2, .5))


def replay(frames, scores, markets, start, end, config, threshold, entries_allowed=True):
    wallet = new_wallet(config)
    execution_config = {**config, 'entry_threshold': threshold}
    market_map = {item['symbol']: item for item in markets}
    rows = {}
    for symbol, raw in frames.items():
        featured = make_features(raw)
        values = scores[symbol]
        for i in range(len(raw) - 1):
            stamp = int(raw.open_time.iloc[i])
            if not start <= stamp < end or not np.isfinite(featured.atr.iloc[i]):
                continue
            price = float(raw.open.iloc[i + 1])
            if int(raw.open_time.iloc[i + 1]) != stamp + INTERVAL_MS:
                raise ValueError('Missing replay execution candle.')
            rows.setdefault(stamp, {})[symbol] = (float(values[i]), float(featured.atr.iloc[i]), price)
    for stamp, batch in sorted(rows.items()):
        signals, quotes = {}, {}
        for symbol, (score, atr, price) in batch.items():
            spread = config.get('backtest_spread_bps', 2) / 20000
            quotes[symbol] = {'bid': price * (1 - spread), 'ask': price * (1 + spread)}
            signals[symbol] = {'symbol': symbol, 'action': 'BUY' if entries_allowed and score >= threshold else 'HOLD',
                               'model_score': score, 'atr': atr, 'candle_time': stamp,
                               'horizon_bars': config['horizon_bars']}
        errors = {} if len(batch) == len(frames) else {'replay': 'Missing synchronized market data.'}
        wallet, _ = paper_cycle(wallet, signals, quotes, market_map, errors, stamp + INTERVAL_MS, execution_config)
        wallet['trade_history'] = wallet['trade_history'][-32:]
        wallet['equity_history'] = wallet['equity_history'][-32:]
    return {'equity': wallet['equity'], 'net_return': wallet['total_pnl'] / wallet['initial_balance'],
            'max_drawdown': wallet['max_drawdown'], 'closed_trades': wallet['closed_count'],
            'wins': wallet['winning_count'], 'fees_paid': wallet['fees_paid'],
            'open_positions': len(wallet['positions']), 'halted': wallet['halted']}


def compare(frames, markets, config):
    train, validation, test = pooled_splits(frames, config['horizon_bars'])
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.04,
                          subsample=.8, colsample_bytree=.9, random_state=42,
                          n_jobs=2, eval_metric='logloss')
    model.fit(train[FEATURES], train.target.astype(int),
              eval_set=[(validation[FEATURES], validation.target.astype(int))], verbose=False)
    policy = select_entry_policy(validation, model.predict_proba(validation[FEATURES])[:, 1], config, markets)
    ml_scores = {symbol: model.predict_proba(make_features(raw)[FEATURES])[:, 1]
                 for symbol, raw in frames.items()}
    breakout_scores = {symbol: trend_scores(raw) for symbol, raw in frames.items()}
    windows = {'validation': (int(validation.open_time.min()), int(validation.open_time.max()) + INTERVAL_MS),
               'test': (int(test.open_time.min()), int(test.open_time.max()) + INTERVAL_MS)}
    results = {}
    for name, scores, threshold in (('model', ml_scores, policy['entry_threshold']),
                                     ('trend_breakout', breakout_scores, .8)):
        results[name] = {part: replay(frames, scores, markets, start, end, config, threshold)
                         for part, (start, end) in windows.items()}
        valid = results[name]['validation']
        results[name]['validation_passed'] = (valid['closed_trades'] >= 20 and valid['net_return'] > 0
                                               and not valid['halted'])
        if name == 'model':
            results[name]['validation_passed'] &= policy['status'] == 'validated'
    eligible = [name for name, result in results.items() if result['validation_passed']]
    selected = max(eligible, key=lambda name: (results[name]['validation']['net_return'],
                                               results[name]['validation']['max_drawdown'])) if eligible else None
    return {'horizon_bars': config['horizon_bars'], 'initial_balance': config['initial_balance_usdt'],
            'markets': len(markets), 'windows': windows, 'threshold_policy': policy,
            'strategies': results, 'selected_on_validation': selected,
            'test_supports_selected': bool(selected and results[selected]['test']['net_return'] > 0
                                            and results[selected]['test']['closed_trades'] >= 20
                                            and not results[selected]['test']['halted']),
            'assumptions': {'execution': 'next open quotes; exits checked only at scheduled observations',
                            'spread_bps': config.get('backtest_spread_bps', 2),
                            'fee_bps_per_side': config['fee_bps_per_side'],
                            'slippage_bps_per_side': config['slippage_bps_per_side'],
                            'external_context': 'excluded: no historical point-in-time news archive',
                            'online_learning': 'excluded: baseline policy comparison only'}}


def publish_result(result):
    store = GitHubDeliveryState(branch='cryptoai-models')
    path = '/contents/research/strategy-comparison.json'
    existing = store.request('GET', path, params={'ref': store.branch})
    if existing.status_code not in (200, 404):
        raise RuntimeError('Cannot read comparison publication state.')
    body = {'branch': store.branch, 'message': 'Publish strategy comparison diagnostics',
            'content': base64.b64encode(json.dumps(result, allow_nan=False).encode()).decode()}
    if existing.status_code == 200:
        body['sha'] = existing.json()['sha']
    if store.request('PUT', path, json=body).status_code not in (200, 201):
        raise RuntimeError('Comparison publication failed.')


def main(publish=False):
    config = load_config()
    candidates = universe(config['universe_size'] + 25)
    now = int(public_data('/time')['serverTime'])
    frames, errors = candle_windows([item['symbol'] for item in candidates], config['training_candles'], now)
    markets = [item for item in candidates if item['symbol'] in frames][:config['universe_size']]
    if len(markets) != config['universe_size']:
        raise ValueError('Insufficient complete markets for comparison.')
    result = compare({item['symbol']: frames[item['symbol']] for item in markets}, markets, config)
    result['excluded_markets'] = errors
    result['data_collected_at'] = datetime.fromtimestamp(now / 1000, timezone.utc).isoformat()
    result['code_revision'] = os.environ.get('GITHUB_SHA')
    print(json.dumps(result, indent=2, allow_nan=False))
    if publish:
        publish_result(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    main(publish=parser.parse_args().publish)
