import json
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.core import DATA, FEATURES, INTERVAL_MS, MODEL_DIR, make_features
from app.market import candle_frame, public_data
from app.portfolio import load_config
from app.strategy_comparison import replay, trend_scores

RETURN_FEATURES = FEATURES + ['return_24', 'return_96', 'atr_relative', 'volume_relative',
                              'hour_sin', 'hour_cos', 'trend_distance']


def return_features(raw):
    frame = make_features(raw)
    for period in (24, 96):
        frame[f'return_{period}'] = frame.close.pct_change(period)
    frame['atr_relative'] = frame.atr_pct / frame.atr_pct.rolling(96).mean()
    frame['volume_relative'] = frame.volume / frame.volume.rolling(96).mean()
    hour = (frame.open_time // INTERVAL_MS % 96) / 96 * 2 * np.pi
    frame['hour_sin'], frame['hour_cos'] = np.sin(hour), np.cos(hour)
    frame['trend_distance'] = (frame.close - frame.close.ewm(span=50, adjust=False).mean()) / frame.atr
    return frame.replace([np.inf, -np.inf], np.nan)


def execution_returns(frame, config):
    horizon = config['horizon_bars']
    fee, slip = config['fee_bps_per_side'] / 10000, config['slippage_bps_per_side'] / 10000
    spread = config.get('backtest_spread_bps', 2) / 20000
    entry = frame.open.shift(-1) * (1 + spread) * (1 + slip)
    stop, target = entry - 1.5 * frame.atr, entry + 3 * frame.atr
    exit_price = pd.Series(np.nan, index=frame.index)
    for step in range(1, horizon + 1):
        bid = frame.open.shift(-(step + 1)) * (1 - spread)
        closed = (bid <= stop) | (bid >= target) | (step == horizon)
        exit_price = exit_price.where(exit_price.notna(), bid.where(closed))
    result = (exit_price * (1 - slip) * (1 - fee) / (entry * (1 + fee)) - 1) * 10000
    result[frame.open.shift(-(horizon + 1)).isna()] = np.nan
    return result


def policy_scores(predictions, featured, policy):
    if policy['kind'] == 'breakout':
        return trend_scores(featured)
    if policy['kind'] == 'pullback':
        enter = (featured.ema_gap > 0) & featured.rsi_14.between(40, 55) & (featured.return_1 <= 0)
        return np.where(enter, .8, .5)
    enter = predictions > policy['minimum_net_bps']
    if policy.get('trend_filter'):
        enter &= (featured.ema_gap > 0).to_numpy()
    exit_signal = predictions < -10 if policy.get('model_exit') else np.zeros(len(featured), dtype=bool)
    return np.where(enter, .8, np.where(exit_signal, .2, .5))


def candidate_policies():
    entries = ([{'kind': 'net_return', 'minimum_net_bps': edge, 'model_exit': exit_model}
                for edge in (0, 10, 20) for exit_model in (False, True)]
               + [{'kind': 'net_return', 'minimum_net_bps': 10, 'trend_filter': True},
                  {'kind': 'breakout'}, {'kind': 'pullback'}])
    return [{**entry, 'holding_bars': holding} for entry in entries for holding in (4, 8, 16)]


def select_candidate(results):
    eligible = [item for item in results if item['validation']['closed_trades'] >= 20
                and item['validation']['net_return'] > 0 and not item['validation']['halted']]
    return max(eligible, key=lambda item: (item['validation']['net_return'],
                                           item['validation']['max_drawdown']), default=None)


def forecast_error(actual, predicted, training_mean):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    usable = np.isfinite(actual) & np.isfinite(predicted)
    if not usable.any():
        raise ValueError('No mature forecast targets.')
    return {'rows': int(usable.sum()),
            'model_rmse_bps': float(np.sqrt(np.mean((actual[usable] - predicted[usable]) ** 2))),
            'training_mean_rmse_bps': float(np.sqrt(np.mean((actual[usable] - training_mean) ** 2)))}


def research(raw, market, config, fresh=None):
    if not (raw.open_time.diff().dropna() == INTERVAL_MS).all():
        raise ValueError('Incomplete historical candle sequence.')
    featured = return_features(raw)
    featured['net_return_bps'] = execution_returns(featured, config)
    a, b = int(raw.open_time.iloc[int(len(raw) * .6)]), int(raw.open_time.iloc[int(len(raw) * .8)])
    label_end = featured.open_time + (config['horizon_bars'] + 1) * INTERVAL_MS
    policies = candidate_policies()
    evaluation_bars = max(item['holding_bars'] for item in policies) + 1
    evaluation_end = featured.open_time + evaluation_bars * INTERVAL_MS
    train = featured[(label_end < a)].dropna(subset=RETURN_FEATURES + ['net_return_bps'])
    validation = featured[(featured.open_time >= a) & (evaluation_end < b)]
    test = featured[(featured.open_time >= b) & (evaluation_end <= raw.open_time.max())]
    model = XGBRegressor(n_estimators=200, max_depth=2, learning_rate=.03,
                         min_child_weight=50, reg_lambda=10, subsample=.8,
                         colsample_bytree=.9, random_state=42, n_jobs=2,
                         objective='reg:squarederror')
    model.fit(train[RETURN_FEATURES], train.net_return_bps)
    predictions = model.predict(featured[RETURN_FEATURES])

    def run(policy, part, full=raw, features=featured, predicted=predictions, costs=config):
        scores = policy_scores(predicted, features, policy)
        return replay({market['symbol']: full}, {market['symbol']: scores}, [market],
                      int(part.open_time.min()), int(part.open_time.max()) + INTERVAL_MS,
                      {**costs, 'horizon_bars': policy['holding_bars']}, .8)

    candidates = [{'policy': policy, 'validation': run(policy, validation)} for policy in policies]
    selected = select_candidate(candidates)
    evidence = [item for item in candidates if item['validation']['closed_trades'] >= 20]
    diagnostic = selected or max(evidence or candidates, key=lambda item: item['validation']['net_return'])
    model_candidate = max((item for item in candidates if item['policy']['kind'] == 'net_return'),
                          key=lambda item: item['validation']['net_return'])
    policy = diagnostic['policy']
    test_result = run(policy, test)
    zero_cost = {**config, 'fee_bps_per_side': 0, 'slippage_bps_per_side': 0, 'backtest_spread_bps': 0}
    cost_stress = {**config, 'fee_bps_per_side': 15, 'slippage_bps_per_side': 4}
    result = {'model_target': 'net bps after scheduled ATR stop/target or one-hour exit and all costs',
              'prediction_horizon_bars': config['horizon_bars'], 'candidate_count': len(candidates), 'candidates': candidates,
              'selected_on_validation': selected, 'diagnostic_policy': policy,
              'test': test_result, 'test_without_costs': run(policy, test, costs=zero_cost),
              'test_higher_costs': run(policy, test, costs=cost_stress),
              'net_return_model_diagnostic': {**model_candidate, 'test': run(model_candidate['policy'], test)},
              'forecast_error': {
                  name: forecast_error(part.net_return_bps, predictions[part.index], train.net_return_bps.mean())
                  for name, part in (('validation', validation), ('test', test))},
              'fresh': None, 'approved_for_forward_trial': False,
              'limitations': ['Historical assumed quotes, not exchange fills.',
                              'No point-in-time news or macro archive.',
                              'Reused historical and post-archive slices are exploratory, not fresh forward confirmation.',
                              'No automated publication or live activation.']}
    if fresh is not None:
        combined = pd.concat([raw, fresh]).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
        if not (combined.open_time.diff().dropna() == INTERVAL_MS).all():
            raise ValueError('Fresh holdout is not continuous with the historical archive.')
        fresh_features = return_features(combined)
        fresh_part = fresh_features[fresh_features.open_time > raw.open_time.max()].iloc[:-evaluation_bars]
        if len(fresh_part) < 100:
            raise ValueError('Insufficient fresh holdout.')
        fresh_predictions = model.predict(fresh_features[RETURN_FEATURES])
        result['fresh'] = run(policy, fresh_part, full=combined, features=fresh_features, predicted=fresh_predictions)
        result['net_return_model_diagnostic']['fresh'] = run(
            model_candidate['policy'], fresh_part, full=combined, features=fresh_features, predicted=fresh_predictions)
        result['fresh_window'] = {'start': int(fresh_part.open_time.min()), 'end': int(fresh_part.open_time.max()),
                                  'rows': len(fresh_part)}
    results_to_pass = [test_result, result['test_higher_costs'], result['fresh']]
    result['approved_for_forward_trial'] = bool(selected and all(
        item and item['closed_trades'] >= 20 and item['net_return'] > 0 and not item['halted']
        for item in results_to_pass))
    return result


def main():
    started = time.monotonic()
    with sqlite3.connect(f'file:{(DATA / "crypto.db").as_posix()}?mode=ro', uri=True) as connection:
        raw = pd.read_sql_query("SELECT * FROM candles WHERE symbol='BTCUSDT' ORDER BY open_time", connection)
    bundle = joblib.load(MODEL_DIR / 'portfolio.joblib')
    market = next(item for item in bundle['markets'] if item['symbol'] == 'BTCUSDT')
    now = int(public_data('/time')['serverTime'])
    count = (now // INTERVAL_MS * INTERVAL_MS - int(raw.open_time.max())) // INTERVAL_MS
    if not 100 < count <= 3000:
        raise ValueError('Fresh holdout must contain 100..3000 candles after the archive.')
    fresh = candle_frame('BTCUSDT', count, now)
    result = research(raw, market, load_config(), fresh)
    result['elapsed_seconds'] = time.monotonic() - started
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
