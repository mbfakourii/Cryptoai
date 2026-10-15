import argparse
import json
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.core import DATA, FEATURES, INTERVAL_MS, MODEL_DIR
from app.market_context import CONTEXT_DIMENSIONS
from app.online_learning import feature_vector, learn, new_classifier, probability
from app.portfolio import load_config
from app.portfolio_models import pooled_splits
from app.strategy_comparison import replay


def sequential_scores(rows, baseline, horizon):
    learner = new_classifier()
    pending, scores = [], []
    for index, row in rows.iterrows():
        stamp = int(row.open_time)
        matured = [item for item in pending if item[0] <= stamp]
        pending = [item for item in pending if item[0] > stamp]
        for _, vector, target in matured:
            learn(learner, [vector], [target], deployed_blend=True)
        base = float(baseline[index])
        vector = feature_vector([float(row[key]) for key in FEATURES],
                                [0.] * CONTEXT_DIMENSIONS, base)
        scores.append(.5 * base + .5 * probability(learner, vector))
        pending.append((stamp + horizon * INTERVAL_MS, vector, int(row.target)))
    return np.asarray(scores)


def threshold_candidates(validation_scores, config):
    values = np.asarray(validation_scores)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite validation scores are required.')
    exit_score = float(np.quantile(values, .25))
    candidates = {(config['entry_threshold'], config['exit_threshold'])}
    for quantile in (.5, .75, .9, .95):
        entry = float(np.quantile(values, quantile))
        if 0 < exit_score < entry < 1:
            candidates.add((entry, exit_score))
    return [{'entry': entry, 'exit': exit_score} for entry, exit_score in sorted(candidates)]


def select_policy(candidates):
    eligible = [item for item in candidates if item['validation']['closed_trades'] >= 20
                and item['validation']['net_return'] > 0 and not item['validation']['halted']]
    return max(eligible, key=lambda item: (item['validation']['net_return'],
                                           item['validation']['max_drawdown']), default=None)


def compare(raw, market, config):
    if not (raw.open_time.diff().dropna() == INTERVAL_MS).all():
        raise ValueError('Replay requires continuous closed candles.')
    symbol = market['symbol']
    train, validation, test = pooled_splits({symbol: raw}, config['horizon_bars'])
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.04,
                          subsample=.8, colsample_bytree=.9, random_state=42,
                          n_jobs=2, eval_metric='logloss')
    model.fit(train[FEATURES], train.target.astype(int),
              eval_set=[(validation[FEATURES], validation.target.astype(int))], verbose=False)
    rows = pd.concat([validation, test]).sort_values('open_time').reset_index(drop=True)
    baseline = model.predict_proba(rows[FEATURES])[:, 1]
    adaptive = sequential_scores(rows, baseline, config['horizon_bars'])
    validation_mask = rows.open_time < test.open_time.min()
    scores = {symbol: raw.open_time.map(dict(zip(rows.open_time, adaptive))).to_numpy()}

    def run(part, policy):
        trial_config = {**config, 'entry_threshold': policy['entry'], 'exit_threshold': policy['exit']}
        return replay({symbol: raw}, scores, [market], int(part.open_time.min()),
                      int(part.open_time.max()) + INTERVAL_MS, trial_config, policy['entry'])

    candidates = [{**policy, 'validation': run(validation, policy)}
                  for policy in threshold_candidates(adaptive[validation_mask], config)]
    selected = select_policy(candidates)
    with_evidence = [item for item in candidates if item['validation']['closed_trades'] >= 20]
    diagnostic = selected or max(with_evidence or candidates,
                                  key=lambda item: item['validation']['net_return'])
    result = run(test, diagnostic)
    original = run(test, {'entry': config['entry_threshold'], 'exit': config['exit_threshold']})
    return {'symbol': symbol, 'horizon_bars': config['horizon_bars'],
            'rows': {'train': len(train), 'validation': len(validation), 'test': len(test)},
            'test_start': int(test.open_time.min()), 'test_end': int(test.open_time.max()),
            'positive_label_fraction': float(test.target.mean()),
            'test_brier': {'baseline': float(np.mean((baseline[~validation_mask] - test.target.to_numpy()) ** 2)),
                           'adaptive': float(np.mean((adaptive[~validation_mask] - test.target.to_numpy()) ** 2))},
            'maximum_adaptive_test_score': float(adaptive[~validation_mask].max()),
            'original_threshold_test': original, 'candidates': candidates,
            'selected_on_validation': selected, 'tested_policy': {'entry': diagnostic['entry'], 'exit': diagnostic['exit']},
            'test_result': result,
            'test_passed': bool(selected and result['closed_trades'] >= 20
                                and result['net_return'] > 0 and not result['halted']),
            'limitations': ['Technical features only; no point-in-time news or macro archive.',
                            'Historical next-open quotes with assumed 2 bps spread, fees and slippage.',
                            'Diagnostic fallback is never approval; test results never select a threshold.',
                            'Previously inspected history is not fresh forward confirmation.']}


def main(symbol):
    started = time.monotonic()
    with sqlite3.connect(f'file:{(DATA / "crypto.db").as_posix()}?mode=ro', uri=True) as connection:
        raw = pd.read_sql_query('SELECT * FROM candles WHERE symbol=? ORDER BY open_time',
                                connection, params=(symbol,))
    bundle = joblib.load(MODEL_DIR / 'portfolio.joblib')
    market = next(item for item in bundle['markets'] if item['symbol'] == symbol)
    result = compare(raw, market, load_config())
    result['elapsed_seconds'] = time.monotonic() - started
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', choices=('BTCUSDT', 'ETHUSDT'), default='BTCUSDT')
    main(parser.parse_args().symbol)
