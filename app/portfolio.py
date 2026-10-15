import copy
import json
import math
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

from app.core import INTERVAL_MS

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'paper-trading.json'


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if config['mode'] != 'paper' or not 0 < config['initial_balance_usdt'] <= 1_000_000:
        raise ValueError('Only a virtual paper wallet is supported.')
    if config.get('paper_entry_policy', 'validated') not in ('validated', 'after_training'):
        raise ValueError('Invalid paper entry policy.')
    if config['universe_size'] != 50 or not 1000 <= config['training_candles'] <= 10000:
        raise ValueError('Invalid training universe configuration.')
    if config['report_interval_minutes'] < 10 or config['report_interval_minutes'] % 10:
        raise ValueError('Reporting interval must be a multiple of 10 minutes.')
    for key in ('max_position_fraction', 'cash_reserve_fraction', 'risk_per_trade_fraction',
                'max_drawdown_fraction', 'entry_threshold', 'exit_threshold'):
        if not isinstance(config[key], (int, float)) or not 0 < config[key] < 1:
            raise ValueError(f'Invalid risk setting: {key}')
    if not 1 <= config['max_positions'] <= 10 or not 1 <= config['horizon_bars'] <= 96:
        raise ValueError('Invalid position limits.')
    if config['exit_threshold'] >= config['entry_threshold']:
        raise ValueError('Exit threshold must be below entry threshold.')
    for key in ('fee_bps_per_side', 'slippage_bps_per_side'):
        if not 0 <= config[key] <= 100:
            raise ValueError('Invalid execution costs.')
    return config


def new_wallet(config):
    initial = float(config['initial_balance_usdt'])
    return {'schema': 1, 'mode': 'paper', 'initial_balance': initial, 'cash': initial,
            'positions': {}, 'realized_pnl': 0.0, 'fees_paid': 0.0,
            'closed_count': 0, 'winning_count': 0, 'seen_candles': {},
            'peak_equity': initial, 'max_drawdown': 0.0, 'halted': False,
            'equity': initial, 'total_pnl': 0.0, 'unrealized_pnl': 0.0,
            'equity_history': [], 'trade_history': [], 'last_cycle_slot': None,
            'last_report_slot': None, 'report_status': 'not_sent', 'updated_at': None}


def validate_wallet(state):
    if state.get('schema') != 1 or state.get('mode') != 'paper':
        raise ValueError('Invalid paper wallet; refusing to reset stored capital.')
    for key in ('cash', 'initial_balance', 'peak_equity', 'fees_paid'):
        if not isinstance(state.get(key), (int, float)) or not math.isfinite(state[key]) or state[key] < 0:
            raise ValueError('Invalid stored wallet balance.')
    for symbol, position in state['positions'].items():
        if position['symbol'] != symbol:
            raise ValueError('Invalid stored position symbol.')
        for key in ('quantity', 'cost', 'last_bid', 'stop', 'target'):
            if not math.isfinite(position[key]) or position[key] <= 0:
                raise ValueError('Invalid stored position values.')


def mark_equity(state, quotes, fee, slip=0):
    liquidation = 0.0
    stale = []
    for symbol, position in state['positions'].items():
        if symbol in quotes:
            position['last_bid'] = quotes[symbol]['bid']
        else:
            stale.append(symbol)
        liquidation += position['quantity'] * position['last_bid'] * (1 - slip) * (1 - fee)
    equity = state['cash'] + liquidation
    state['equity'] = equity
    state['total_pnl'] = equity - state['initial_balance']
    state['unrealized_pnl'] = state['total_pnl'] - state['realized_pnl']
    state['peak_equity'] = max(state['peak_equity'], equity)
    drawdown = equity / state['peak_equity'] - 1
    state['max_drawdown'] = min(state['max_drawdown'], drawdown)
    state['stale_positions'] = stale
    return drawdown


def rounded_quantity(notional, price, market):
    step = Decimal(market['step_size'])
    if step <= 0:
        raise ValueError('Invalid quantity step.')
    raw = Decimal(str(notional)) / Decimal(str(price))
    quantity = (raw / step).to_integral_value(rounding=ROUND_DOWN) * step
    if not Decimal(market['min_qty']) <= quantity <= Decimal(market['max_qty']):
        return 0.0
    return float(quantity)


def paper_cycle(wallet, signals, quotes, markets, errors, now, config):
    validate_wallet(wallet)
    state = copy.deepcopy(wallet)
    slot = now // 600_000
    if state['last_cycle_slot'] is not None and slot <= state['last_cycle_slot']:
        return state, []
    fee = config['fee_bps_per_side'] / 10000
    slip = config['slippage_bps_per_side'] / 10000
    for quote in quotes.values():
        if not all(math.isfinite(quote[key]) for key in ('bid', 'ask')) or not 0 < quote['bid'] <= quote['ask']:
            raise ValueError('Invalid execution quotes.')
    events, closed = [], set()
    for symbol, position in list(state['positions'].items()):
        if symbol not in quotes:
            continue
        bid = quotes[symbol]['bid']
        signal = signals.get(symbol)
        reason = ('SL' if bid <= position['stop'] else 'TP' if bid >= position['target']
                  else 'TIME' if now >= position['expires_at']
                  else 'MODEL' if signal and signal['action'] in ('BUY', 'HOLD')
                  and signal['candle_time'] > position['signal_time']
                  and signal['model_score'] < config['exit_threshold'] else None)
        if not reason:
            continue
        price = bid * (1 - slip)
        gross = position['quantity'] * price
        commission = gross * fee
        proceeds = gross - commission
        pnl = proceeds - position['cost']
        state['cash'] += proceeds
        state['realized_pnl'] += pnl
        state['fees_paid'] += commission
        state['closed_count'] += 1
        state['winning_count'] += int(pnl > 0)
        events.append({'side': 'SELL', 'symbol': symbol, 'time': now, 'price': price,
                       'quantity': position['quantity'], 'notional': gross,
                       'fee': commission, 'pnl': pnl, 'reason': reason,
                       'entry_time': position['entered_at'], 'entry_price': position['entry_price']})
        del state['positions'][symbol]
        closed.add(symbol)
    drawdown = mark_equity(state, quotes, fee, slip)
    if drawdown <= -config['max_drawdown_fraction']:
        state['halted'] = True
    statuses = {symbol: signal['action'] for symbol, signal in signals.items()}
    candidates = sorted(signals.values(), key=lambda signal: (-signal.get('model_score', 0), signal['symbol']))
    for signal in candidates:
        symbol = signal['symbol']
        if signal['action'] not in ('BUY', 'HOLD'):
            continue
        candle = signal['candle_time']
        if candle <= state['seen_candles'].get(symbol, -1):
            continue
        state['seen_candles'][symbol] = candle
        if (signal['action'] != 'BUY' or signal['model_score'] < config['entry_threshold']
                or symbol in state['positions'] or symbol in closed):
            continue
        if errors or state['stale_positions'] or state['halted']:
            statuses[symbol] = 'RISK_PAUSED'
            continue
        if len(state['positions']) >= config['max_positions']:
            statuses[symbol] = 'POSITION_LIMIT'
            continue
        market = markets.get(symbol)
        quote = quotes.get(symbol)
        if not market or not quote:
            statuses[symbol] = 'MARKET_UNAVAILABLE'
            continue
        price = quote['ask'] * (1 + slip)
        atr = signal['atr']
        stop = price - 1.5 * atr
        if not math.isfinite(atr) or atr <= 0 or stop <= 0:
            raise ValueError('Invalid stop distance.')
        equity = state['equity']
        available = max(0, state['cash'] - equity * config['cash_reserve_fraction'])
        notional = min(available / (1 + fee), equity * config['max_position_fraction'],
                       equity * config['risk_per_trade_fraction'] / (1.5 * atr / price))
        quantity = rounded_quantity(notional, price, market)
        gross = quantity * price
        if quantity <= 0 or gross < market['min_notional']:
            statuses[symbol] = 'BELOW_MINIMUM_ORDER'
            continue
        commission = gross * fee
        cost = gross + commission
        if cost > state['cash']:
            raise ValueError('Paper order would exceed available cash.')
        state['cash'] -= cost
        state['fees_paid'] += commission
        state['positions'][symbol] = {'symbol': symbol, 'quantity': quantity,
                                      'cost': cost, 'entry_price': price, 'last_bid': quote['bid'],
                                      'stop': stop, 'target': price + 3 * atr, 'entered_at': now,
                                      'expires_at': now + signal['horizon_bars'] * INTERVAL_MS,
                                      'signal_time': candle, 'model_score': signal['model_score']}
        events.append({'side': 'BUY', 'symbol': symbol, 'time': now, 'price': price,
                       'quantity': quantity, 'notional': gross, 'fee': commission,
                       'pnl': None, 'reason': 'MODEL'})
        statuses[symbol] = 'BOUGHT'
        mark_equity(state, quotes, fee, slip)
    if mark_equity(state, quotes, fee, slip) <= -config['max_drawdown_fraction']:
        state['halted'] = True
    state['last_cycle_slot'] = slot
    state['updated_at'] = now
    state['signals'] = statuses
    state['market_errors'] = errors
    state['last_events'] = events
    state['trade_history'] = (state['trade_history'] + events)[-1000:]
    state['equity_history'] = (state['equity_history'] + [{'time': now, 'equity': state['equity']}])[-2048:]
    validate_wallet(state)
    return state, events
