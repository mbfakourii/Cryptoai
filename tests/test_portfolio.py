import copy

import pytest

from app.portfolio import load_config, new_wallet, paper_cycle
from app.portfolio_runner import persist_and_report


def fixtures():
    config = load_config()
    signals = {symbol: {'symbol': symbol, 'action': 'BUY', 'model_score': .8,
                        'candle_time': 900000, 'atr': .01, 'horizon_bars': 12}
               for symbol in ('AAAUSDT', 'BBBUSDT', 'CCCUSDT', 'DDDUSDT')}
    quotes = {symbol: {'bid': 1., 'ask': 1.001} for symbol in signals}
    markets = {symbol: {'step_size': '.01', 'min_qty': '.01', 'max_qty': '100000',
                        'min_notional': 5.} for symbol in signals}
    return config, signals, quotes, markets


def test_orders_share_twenty_dollars_and_preserve_cash_reserve():
    config, signals, quotes, markets = fixtures()
    state, events = paper_cycle(new_wallet(config), signals, quotes, markets, {}, 1800000, config)
    assert len(events) == 2
    assert state['cash'] >= state['equity'] * config['cash_reserve_fraction']
    assert sum(position['cost'] for position in state['positions'].values()) + state['cash'] == pytest.approx(20)
    assert state['equity'] < 20


def test_restart_cannot_buy_same_closed_candle_again():
    config, signals, quotes, markets = fixtures()
    state, _ = paper_cycle(new_wallet(config), signals, quotes, markets, {}, 1800000, config)
    duplicate, events = paper_cycle(state, signals, quotes, markets, {}, 1800001, config)
    assert duplicate == state and not events
    later, events = paper_cycle(state, signals, quotes, markets, {}, 2400000, config)
    assert not events
    assert later['cash'] == state['cash']


def test_minimum_order_is_respected():
    config, signals, quotes, markets = fixtures()
    for market in markets.values():
        market['min_notional'] = 10
    state, events = paper_cycle(new_wallet(config), signals, quotes, markets, {}, 1800000, config)
    assert not events and state['cash'] == 20
    assert set(state['signals'].values()) == {'BELOW_MINIMUM_ORDER'}


def test_market_error_blocks_new_buys_but_stop_loss_still_sells():
    config, signals, quotes, markets = fixtures()
    state, _ = paper_cycle(new_wallet(config), signals, quotes, markets, {}, 1800000, config)
    for symbol in state['positions']:
        quotes[symbol] = {'bid': .8, 'ask': .81}
    next_state, events = paper_cycle(state, signals, quotes, markets, {'DDDUSDT': 'unavailable'}, 2400000, config)
    assert all(event['side'] == 'SELL' and event['reason'] == 'SL' for event in events)
    assert not next_state['positions']
    assert next_state['total_pnl'] == pytest.approx(next_state['realized_pnl'])
    assert next_state['fees_paid'] > state['fees_paid']


def test_drawdown_halts_entries_persistently():
    config, signals, quotes, markets = fixtures()
    wallet = new_wallet(config)
    wallet['peak_equity'] = 30
    state, events = paper_cycle(wallet, signals, quotes, markets, {}, 1800000, config)
    assert state['halted'] and not events


class Store:
    def __init__(self, conflict=False):
        self.states = []
        self.conflict = conflict

    def write(self, state, sha):
        if self.conflict:
            raise RuntimeError('Conflict')
        self.states.append(copy.deepcopy(state))
        return 'new-sha'


def test_report_reservation_survives_unknown_delivery_and_suppresses_retry():
    config = load_config()
    state = new_wallet(config)
    state['updated_at'] = 1800000
    store = Store()
    calls = []

    def fail(state):
        calls.append(state)
        raise RuntimeError('Unknown outcome')

    with pytest.raises(RuntimeError):
        persist_and_report(store, state, None, config, fail)
    assert store.states[0]['report_status'] == 'reserved'
    assert store.states[-1]['report_status'] == 'unknown_or_failed'
    persist_and_report(store, state, 'new-sha', config, fail)
    assert len(calls) == 1


def test_conflicting_wallet_update_never_sends_report():
    config = load_config()
    state = new_wallet(config)
    state['updated_at'] = 1800000
    calls = []
    with pytest.raises(RuntimeError):
        persist_and_report(Store(True), state, None, config, calls.append)
    assert not calls
