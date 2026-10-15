import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier, XGBRegressor

from app.btc_strategy_research import RETURN_FEATURES, execution_returns, policy_scores, return_features
from app.core import DATA, FEATURES, INTERVAL_MS
from app.market import candle_frame, public_data
from app.portfolio import load_config, new_wallet, paper_cycle
from app.single_market_replay import sequential_scores, threshold_candidates
from app.strategy_comparison import trend_scores

DAY_MS = 96 * INTERVAL_MS


def windows(start, end, train_days=180, validation_days=30, test_days=60):
    if min(train_days, validation_days, test_days) <= 0:
        raise ValueError('Window lengths must be positive.')
    test_start = start + (train_days + validation_days) * DAY_MS
    while test_start + 30 * DAY_MS <= end:
        yield {'train_start': test_start - (train_days + validation_days) * DAY_MS,
               'validation_start': test_start - validation_days * DAY_MS,
               'test_start': test_start, 'test_end': min(test_start + test_days * DAY_MS, end)}
        test_start += test_days * DAY_MS


def validate_history(raw, start, end):
    numeric = raw[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    if (len(raw) != (end - start) // INTERVAL_MS or raw.empty
            or raw.open_time.iloc[0] != start or raw.open_time.iloc[-1] != end - INTERVAL_MS
            or not (raw.open_time.diff().dropna() == INTERVAL_MS).all()
            or not np.isfinite(numeric.to_numpy()).all()
            or (raw[['open', 'high', 'low', 'close']] <= 0).any().any()
            or (raw.volume < 0).any()
            or (raw.high < raw[['open', 'close', 'low']].max(axis=1)).any()
            or (raw.low > raw[['open', 'close', 'high']].min(axis=1)).any()):
        raise ValueError('Expected complete, ordered, valid closed candles for the requested range.')


def load_history(start, end, cache):
    if cache.exists():
        raw = pd.read_csv(cache)
        validate_history(raw, start, end)
        return raw
    cache.parent.mkdir(parents=True, exist_ok=True)
    def load_chunk(chunk_start):
        chunk_end = min(chunk_start + 30 * DAY_MS, end)
        part = cache.parent / f'btc-{chunk_start}-{chunk_end}.csv'
        if part.exists():
            chunk = pd.read_csv(part)
        else:
            for attempt in range(3):
                try:
                    chunk = candle_frame('BTCUSDT', (chunk_end - chunk_start) // INTERVAL_MS, chunk_end)
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            validate_history(chunk, chunk_start, chunk_end)
            chunk.to_csv(part, index=False)
        validate_history(chunk, chunk_start, chunk_end)
        print(f'History ready through {pd.to_datetime(chunk_end, unit="ms", utc=True).date()}',
              file=sys.stderr, flush=True)
        return chunk

    with ThreadPoolExecutor(max_workers=3) as pool:
        chunks = list(pool.map(load_chunk, range(start, end, 30 * DAY_MS)))
    raw = pd.concat(chunks, ignore_index=True)
    validate_history(raw, start, end)
    raw.to_csv(cache, index=False)
    return raw


def replay_window(featured, scores, market, start, end, policy, config, entries_allowed=True):
    execution = {**config, 'entry_threshold': policy['entry'], 'exit_threshold': policy['exit']}
    wallet = new_wallet(execution)
    times = featured.open_time.to_numpy(dtype=np.int64)
    prices, atrs = featured.open.to_numpy(), featured.atr.to_numpy()
    entry_end = end - (policy['holding_bars'] + 2) * INTERVAL_MS
    spread = config.get('backtest_spread_bps', 2) / 20000
    indices = np.flatnonzero((times >= start) & (times + INTERVAL_MS < end))
    symbol = market['symbol']
    for i in indices:
        if i + 1 >= len(times) or times[i + 1] != times[i] + INTERVAL_MS:
            raise ValueError('Missing execution candle.')
        score = float(scores[i])
        if not np.isfinite(score) or not np.isfinite(atrs[i]):
            raise ValueError('Missing replay prediction or ATR.')
        signal = {'symbol': symbol,
                  'action': 'BUY' if entries_allowed and times[i] < entry_end and score >= policy['entry'] else 'HOLD',
                  'model_score': score, 'atr': float(atrs[i]), 'candle_time': int(times[i]),
                  'horizon_bars': policy['holding_bars']}
        quotes = {symbol: {'bid': float(prices[i + 1]) * (1 - spread),
                           'ask': float(prices[i + 1]) * (1 + spread)}}
        wallet, _ = paper_cycle(wallet, {symbol: signal}, quotes, {symbol: market}, {},
                                int(times[i + 1]), execution)
        wallet['trade_history'] = wallet['trade_history'][-32:]
        wallet['equity_history'] = wallet['equity_history'][-32:]
    if wallet['positions']:
        raise ValueError('Replay ended before positions expired.')
    return {'net_return': wallet['total_pnl'] / wallet['initial_balance'],
            'closed_trades': wallet['closed_count'], 'wins': wallet['winning_count'],
            'max_drawdown': wallet['max_drawdown'], 'fees_paid': wallet['fees_paid'],
            'halted': wallet['halted']}


def choose_on_validation(candidates):
    eligible = [item for item in candidates if item['validation']['closed_trades'] >= 20
                and item['validation']['net_return'] > 0 and not item['validation']['halted']]
    evidence = [item for item in candidates if item['validation']['closed_trades'] >= 20]
    chosen = max(eligible or evidence or candidates, key=lambda item: item['validation']['net_return'])
    return chosen, bool(eligible)


def summarize(folds):
    summary = {}
    for family in folds[0]['strategies']:
        results = [fold['strategies'][family] for fold in folds]
        tests = [item['test'] for item in results]
        gated = [item['test'] for item in results if item['validation_passed']]
        summary[family] = {
            'test_folds': len(tests), 'profitable_test_folds': sum(item['net_return'] > 0 for item in tests),
            'closed_trades': sum(item['closed_trades'] for item in tests),
            'mean_test_return': float(np.mean([item['net_return'] for item in tests])),
            'median_test_return': float(np.median([item['net_return'] for item in tests])),
            'worst_test_drawdown': min(item['max_drawdown'] for item in tests),
            'normalized_compounded_return': float(np.prod([1 + item['net_return'] for item in tests]) - 1),
            'validation_passed_folds': len(gated),
            'gated_closed_trades': sum(item['closed_trades'] for item in gated),
            'gated_normalized_compounded_return': float(np.prod([1 + item['net_return'] for item in gated]) - 1),
            'higher_cost_compounded_return': float(np.prod([1 + item['test_higher_costs']['net_return']
                                                            for item in results]) - 1)}
    ranked = sorted(summary, key=lambda name: summary[name]['normalized_compounded_return'], reverse=True)
    return {'strategies': summary, 'diagnostic_ranking': ranked,
            'ranking_is_not_live_approval': True}


def compare(raw, market, config, progress=None):
    start, end = int(raw.open_time.iloc[0]), int(raw.open_time.iloc[-1]) + INTERVAL_MS
    validate_history(raw, start, end)
    featured = return_features(raw)
    horizon = config['horizon_bars']
    featured['net_return_bps'] = execution_returns(featured, config)
    target_exit = featured.close.shift(-horizon)
    featured['target'] = ((target_exit / featured.open.shift(-1) - 1) > .004).astype(float)
    featured.loc[target_exit.isna(), 'target'] = np.nan
    maturity = featured.open_time + (horizon + 1) * INTERVAL_MS
    folds = []
    for number, bounds in enumerate(windows(start, end), 1):
        train = featured[(featured.open_time >= bounds['train_start'])
                         & (maturity < bounds['validation_start'])].dropna(subset=RETURN_FEATURES + ['target', 'net_return_bps'])
        subsequent = featured[(featured.open_time >= bounds['validation_start'])
                              & (featured.open_time < bounds['test_end'])].copy()
        classifier = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.04,
                                   subsample=.8, colsample_bytree=.9, random_state=42,
                                   n_jobs=2, eval_metric='logloss')
        classifier.fit(train[FEATURES], train.target.astype(int))
        base = classifier.predict_proba(subsequent[FEATURES])[:, 1]
        online_rows = subsequent.copy().reset_index(drop=True)
        online_rows['target'] = online_rows.target.fillna(0)
        adaptive = sequential_scores(online_rows, base, horizon)
        regressor = XGBRegressor(n_estimators=200, max_depth=2, learning_rate=.03,
                                 min_child_weight=50, reg_lambda=10, subsample=.8,
                                 colsample_bytree=.9, random_state=42, n_jobs=2,
                                 objective='reg:squarederror')
        regressor.fit(train[RETURN_FEATURES], train.net_return_bps)
        predicted_returns = regressor.predict(subsequent[RETURN_FEATURES])
        validation_mask = (subsequent.open_time < bounds['test_start']).to_numpy()
        candidates = {}
        for family, values in (('base_classifier', base), ('adaptive_classifier', adaptive)):
            candidates[family] = [({**policy, 'holding_bars': holding}, values)
                                  for policy in threshold_candidates(values[validation_mask], config)
                                  for holding in (4, 8, 16)]
        candidates['net_return'] = [({'kind': 'net_return', 'minimum_net_bps': edge,
                                      'entry': .8, 'exit': .4, 'holding_bars': holding},
                                     policy_scores(predicted_returns, subsequent,
                                                   {'kind': 'net_return', 'minimum_net_bps': edge}))
                                    for edge in (0, 10, 20) for holding in (4, 8, 16)]
        trend = trend_scores(featured)[subsequent.index]
        candidates['trend_breakout'] = [({'entry': .8, 'exit': .4, 'holding_bars': holding}, trend)
                                        for holding in (4, 8, 16)]
        fold = {'number': number, 'bounds_utc': {key: pd.to_datetime(value, unit='ms', utc=True).isoformat()
                                                for key, value in bounds.items()},
                'training_rows': len(train), 'strategies': {}}
        first, last = subsequent.index[0], subsequent.index[-1]
        execution_frame = featured.iloc[first:last + 2].reset_index(drop=True)
        for family, policies in candidates.items():
            evaluations = []
            for index, (policy, values) in enumerate(policies):
                scores = np.append(values, values[-1])
                validation = replay_window(execution_frame, scores, market, bounds['validation_start'],
                                             bounds['test_start'], policy, config)
                evaluations.append({'candidate': index, 'policy': policy, 'validation': validation})
            chosen, passed = choose_on_validation(evaluations)
            policy, values = policies[chosen['candidate']]
            scores = np.append(values, values[-1])
            normal = replay_window(execution_frame, scores, market, bounds['test_start'],
                                   bounds['test_end'], policy, config)
            stress = replay_window(execution_frame, scores, market, bounds['test_start'], bounds['test_end'],
                                   policy, {**config, 'fee_bps_per_side': 15, 'slippage_bps_per_side': 4})
            fold['strategies'][family] = {'policy': policy, 'validation_passed': passed,
                                          'validation_candidates': evaluations, 'test': normal,
                                          'test_higher_costs': stress}
        folds.append(fold)
        if progress:
            progress(fold)
    if not folds:
        raise ValueError('Not enough history for a full walk-forward test.')
    return {'symbol': market['symbol'], 'rows': len(raw), 'folds': folds, 'summary': summarize(folds),
            'config': config, 'market_rules': market,
            'design': {'training_days': 180, 'validation_days': 30, 'test_days': 60,
                       'prediction_horizon_bars': horizon, 'holding_caps_bars': [4, 8, 16],
                       'minimum_validation_trades': 20, 'seed': 42},
            'limitations': ['BTC-only technical proxies; not the deployed pooled model or its news/macro context.',
                            'Scheduled next-open quote approximation, not actual exchange fills.',
                            'Current lot/minimum-order rules are assumed throughout history.',
                            'Historical research includes previously inspected dates; not fresh forward confirmation.',
                            'Each fold starts with the configured wallet balance; compounded returns normalize independent folds.',
                            'Failed validation is reported diagnostically; the gated policy stays in cash for that fold.',
                            'No online model replacement, wallet mutation, or automatic trading activation.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2024-10-01')
    parser.add_argument('--end', default='2026-10-01')
    parser.add_argument('--output', type=Path, default=DATA / 'research' / 'btc-walk-forward.json')
    args = parser.parse_args()
    start, end = [int(pd.Timestamp(value, tz='UTC').timestamp() * 1000) for value in (args.start, args.end)]
    if start >= end or end > int(public_data('/time')['serverTime']) // INTERVAL_MS * INTERVAL_MS:
        raise ValueError('Dates must describe a past closed-candle range.')
    cache = DATA / 'research' / f'btc-{args.start}-{args.end}.csv'
    raw = load_history(start, end, cache)
    info = public_data('/exchangeInfo', {'symbol': 'BTCUSDT'})['symbols'][0]
    filters = {item['filterType']: item for item in info['filters']}
    lot = filters['LOT_SIZE']
    minimum = filters.get('NOTIONAL', filters.get('MIN_NOTIONAL'))
    market = {'symbol': 'BTCUSDT', 'step_size': lot['stepSize'], 'min_qty': lot['minQty'],
              'max_qty': lot['maxQty'], 'min_notional': float(minimum['minNotional'])}

    def progress(fold):
        print(json.dumps({'fold': fold['number'], 'tests': {name: item['test']['net_return']
                          for name, item in fold['strategies'].items()}}), file=sys.stderr, flush=True)

    result = compare(raw, market, load_config(), progress)
    result['source'] = {'provider': 'Binance public spot klines', 'start_inclusive': args.start,
                        'end_exclusive': args.end, 'csv_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}
    result['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'xgboost': xgboost.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
