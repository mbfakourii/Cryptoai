from collections import Counter

import numpy as np

from app.core import INTERVAL_MS
from app.portfolio import new_wallet, paper_cycle

EVALUATION_POLICY = 'shared_wallet_v1'


def shared_wallet_results(frame, scores, threshold, config, markets):
    scores = np.asarray(scores, dtype=float)
    if (scores.shape != (len(frame),) or not len(scores) or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError('Invalid evaluation scores.')
    required = {'symbol', 'open_time', 'open', 'atr'}
    if not required.issubset(frame.columns):
        raise ValueError('Missing evaluation columns.')
    market_map = {market['symbol']: market for market in markets}
    if len(market_map) != len(markets) or set(frame.symbol) != set(market_map):
        raise ValueError('Missing synchronized market metadata.')
    prepared = frame[list(required)].copy()
    prepared['score'] = scores
    batches, times = {}, None
    for symbol, rows in prepared.groupby('symbol', sort=True):
        rows = rows.sort_values('open_time')
        values = rows[['open_time', 'open', 'atr', 'score']].to_numpy(dtype=float)
        stamps = values[:, 0]
        if (not np.isfinite(values).all() or (values[:, 1:3] <= 0).any()
                or (stamps % INTERVAL_MS != 0).any()
                or (np.diff(stamps) != INTERVAL_MS).any()):
            raise ValueError(f'Invalid or missing synchronized candles for {symbol}.')
        if times is not None and not np.array_equal(stamps, times):
            raise ValueError('Missing synchronized market data.')
        times = stamps.astype(np.int64)
        batches[symbol] = values
    wallet = new_wallet(config)
    execution_config = {**config, 'entry_threshold': threshold}
    half_spread = config.get('backtest_spread_bps', 2) / 20000
    returns, blocked = [], Counter()
    for index in range(len(times) - 1):
        signals, quotes = {}, {}
        entry_has_exit = index + 1 + config['horizon_bars'] < len(times)
        for symbol, values in batches.items():
            _, _, atr, score = values[index]
            price = values[index + 1, 1]
            quotes[symbol] = {'bid': price * (1 - half_spread), 'ask': price * (1 + half_spread)}
            signals[symbol] = {'symbol': symbol, 'action': 'BUY' if entry_has_exit and score >= threshold else 'HOLD',
                               'model_score': score, 'atr': atr, 'candle_time': int(times[index]),
                               'horizon_bars': config['horizon_bars']}
        wallet, events = paper_cycle(wallet, signals, quotes, market_map, {},
                                     int(times[index + 1]), execution_config)
        for event in events:
            if event['side'] == 'SELL':
                entry_cost = event['notional'] - event['fee'] - event['pnl']
                returns.append(event['pnl'] / entry_cost)
        blocked.update(status for status in wallet['signals'].values()
                       if status not in ('BUY', 'HOLD', 'BOUGHT'))
        wallet['trade_history'] = wallet['trade_history'][-32:]
        wallet['equity_history'] = wallet['equity_history'][-32:]
    complete = not wallet['positions']
    return {'evaluation_policy': EVALUATION_POLICY, 'synchronized': True, 'completed': complete,
            'start_time': int(times[0]), 'end_time': int(times[-1]),
            'initial_balance': wallet['initial_balance'], 'equity': wallet['equity'],
            'net_return': wallet['total_pnl'] / wallet['initial_balance'],
            'realized_pnl': wallet['realized_pnl'], 'fees_paid': wallet['fees_paid'],
            'max_drawdown': wallet['max_drawdown'], 'halted': wallet['halted'],
            'open_positions': len(wallet['positions']), 'trade_count': wallet['closed_count'],
            'mean_net_return': float(np.mean(returns)) if returns else None,
            'net_profitable_fraction': wallet['winning_count'] / len(returns) if returns else None,
            'blocked_entries': dict(blocked)}


def evaluation_passed(result, minimum_trades=20):
    result = result or {}
    net_return = result.get('net_return')
    return bool(result.get('evaluation_policy') == EVALUATION_POLICY
            and result.get('synchronized') is True and result.get('completed') is True
            and result.get('open_positions') == 0 and result.get('halted') is False
            and result.get('trade_count', 0) >= minimum_trades
            and isinstance(net_return, (int, float)) and np.isfinite(net_return) and net_return > 0)
