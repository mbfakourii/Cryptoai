import copy
import json

import numpy as np
import pandas as pd
import pytest

from app.core import INTERVAL_MS
from app.market_context import CONTEXT_DIMENSIONS
from app.online_learning import (DIMENSIONS, active, adapt_signals, feature_vector,
                                 learn, mature_samples, new_classifier, new_learning_state,
                                 probability, record_trade_outcomes)
from app.portfolio import load_config, new_wallet, paper_cycle
from app.portfolio_runner import WalletStore


def fixtures(count=100):
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .002, count)))
    frame = pd.DataFrame({'open_time': np.arange(count) * INTERVAL_MS, 'open': close,
                          'close': close, 'high': close * 1.01, 'low': close * .99,
                          'volume': rng.uniform(10, 100, count)})
    candle = int(frame.open_time.iloc[-1])
    signal = {'symbol': 'AAAUSDT', 'action': 'BUY', 'model_score': .8,
              'candle_time': candle, 'atr': 1., 'horizon_bars': 12}
    context = {'ready': True, 'vector': [0.] * CONTEXT_DIMENSIONS}
    quotes = {'AAAUSDT': {'bid': float(close[-1]), 'ask': float(close[-1]) + .01}}
    return frame, {'AAAUSDT': signal}, context, quotes


def test_new_learner_starts_with_base_prediction_and_learns_incrementally():
    model = new_classifier()
    vector = feature_vector([0.] * 9, [0.] * CONTEXT_DIMENSIONS, .8)
    assert probability(model, vector) == pytest.approx(.8, abs=1e-6)
    learn(model, [vector] * 10, [0] * 10)
    assert probability(model, vector) < .8
    assert model['samples'] == 10
    restored = json.loads(json.dumps(model))
    learn(restored, [vector], [0])
    assert restored['samples'] == 11 and restored['steps'] > model['steps']


def test_new_candle_is_recorded_once_and_input_state_is_not_mutated():
    frame, signals, context, quotes = fixtures()
    now = 100 * INTERVAL_MS + 30000
    state, _, _ = adapt_signals(None, {'AAAUSDT': frame}, signals, quotes, context, now, load_config())
    original = copy.deepcopy(state)
    repeated, decisions, _ = adapt_signals(state, {'AAAUSDT': frame}, signals, quotes,
                                          context, now + 10000, load_config())
    assert len(repeated['pending']) == 1 and decisions['AAAUSDT']['action'] == 'WAITING'
    assert state == original and state['market']['samples'] == 0


def test_future_return_is_not_learned_until_the_horizon_candle_closes():
    frame, signals, context, quotes = fixtures()
    state, _, _ = adapt_signals(None, {'AAAUSDT': frame}, signals, quotes, context,
                                100 * INTERVAL_MS, load_config())
    future = frame.iloc[[-1]].copy()
    future['open_time'] = 111 * INTERVAL_MS
    future['close'] = quotes['AAAUSDT']['ask'] * 1.01
    mature_samples(state, {'AAAUSDT': future}, 112 * INTERVAL_MS - 1)
    assert state['market']['samples'] == 0
    mature_samples(state, {'AAAUSDT': future}, 112 * INTERVAL_MS)
    assert state['market']['samples'] == 1 and not state['pending']
    mature_samples(state, {'AAAUSDT': future}, 113 * INTERVAL_MS)
    assert state['market']['samples'] == 1


def test_newer_news_does_not_replace_an_older_symbols_frozen_context():
    frame, signals, context, quotes = fixtures()
    now = 100 * INTERVAL_MS
    state, _, _ = adapt_signals(None, {'AAAUSDT': frame}, signals, quotes, context, now, load_config())
    old_context = state['pending'][0]['context_id']
    context['vector'] = [1.] * CONTEXT_DIMENSIONS
    other = {'BBBUSDT': {**signals['AAAUSDT'], 'symbol': 'BBBUSDT'}}
    state, _, _ = adapt_signals(state, {'BBBUSDT': frame}, other,
                                {'BBBUSDT': quotes['AAAUSDT']}, context, now + 1, load_config())
    assert state['contexts'][old_context] == [0.] * CONTEXT_DIMENSIONS
    assert state['pending'][1]['context_id'] != old_context


def test_missing_context_blocks_buys_but_allows_existing_stop_loss():
    frame, signals, context, quotes = fixtures()
    config = load_config()
    wallet = new_wallet(config)
    wallet['positions']['AAAUSDT'] = {'symbol': 'AAAUSDT', 'quantity': .05, 'cost': 5.,
                                     'last_bid': 100., 'stop': 110., 'target': 120.,
                                     'entered_at': 1, 'entry_price': 100., 'expires_at': 200 * INTERVAL_MS,
                                     'signal_time': 0}
    context['ready'] = False
    _, decisions, _ = adapt_signals(None, {'AAAUSDT': frame}, signals, quotes,
                                     context, 100 * INTERVAL_MS, config)
    assert decisions['AAAUSDT']['action'] == 'HOLD'
    _, events = paper_cycle(wallet, decisions, quotes, {}, {}, 100 * INTERVAL_MS, config)
    assert len(events) == 1 and events[0]['reason'] == 'SL'


def test_trade_feedback_uses_net_realized_result_and_entry_features_only_once():
    state = new_learning_state(12)
    snapshot = {'AAAUSDT': {'vector': [1.] * DIMENSIONS, 'feedback_score': .5}}
    buy = {'side': 'BUY', 'symbol': 'AAAUSDT', 'time': 10}
    sell = {'side': 'SELL', 'symbol': 'AAAUSDT', 'time': 20, 'entry_time': 10, 'pnl': -.03}
    record_trade_outcomes(state, [buy], snapshot)
    assert state['feedback']['samples'] == 0
    record_trade_outcomes(state, [sell], {})
    assert state['feedback']['samples'] == 1
    assert probability(state['feedback'], snapshot['AAAUSDT']['vector']) < .5
    record_trade_outcomes(state, [sell], {})
    assert state['feedback']['samples'] == 1


def test_trades_without_historical_entry_features_are_not_fabricated():
    state = new_learning_state(12)
    record_trade_outcomes(state, [{'side': 'SELL', 'symbol': 'OLDUSDT', 'entry_time': 1,
                                  'time': 2, 'pnl': -1.}], {})
    assert state['feedback']['samples'] == 0


def test_adaptive_model_requires_enough_out_of_sample_results_and_better_brier():
    model = new_classifier()
    model.update(samples=200, evaluated=199, brier=.1, baseline_brier=.2)
    assert not active(model, 200)
    model['evaluated'] = 200
    assert active(model, 200)
    model['brier'] = .3
    assert not active(model, 200)


def test_missing_future_candles_are_not_replaced_by_a_different_close():
    frame, signals, context, quotes = fixtures()
    state, _, _ = adapt_signals(None, {'AAAUSDT': frame}, signals, quotes, context,
                                100 * INTERVAL_MS, load_config())
    mature_samples(state, {'AAAUSDT': frame}, 120 * INTERVAL_MS)
    assert state['market']['samples'] == 0 and len(state['pending']) == 1


def test_qualified_online_model_changes_scores_using_external_context():
    frame, signals, context, quotes = fixtures()
    state = new_learning_state(12)
    state['market'].update(samples=200, evaluated=200, brier=.1, baseline_brier=.2)
    state['market']['weights'][-1] = -8.
    context['vector'][-1] = 1.
    _, decisions, _ = adapt_signals(state, {'AAAUSDT': frame}, signals, quotes,
                                     context, 100 * INTERVAL_MS, load_config())
    assert decisions['AAAUSDT']['model_score'] < .5
    assert decisions['AAAUSDT']['action'] == 'HOLD'
    assert decisions['AAAUSDT']['base_model_score'] == .8


def test_qualified_trade_feedback_vetoes_entry_without_changing_exit_score():
    frame, signals, context, quotes = fixtures()
    state = new_learning_state(12)
    state['feedback'].update(samples=30, evaluated=30, brier=.1, baseline_brier=.25, intercept=-4.)
    _, decisions, _ = adapt_signals(state, {'AAAUSDT': frame}, signals, quotes,
                                     context, 100 * INTERVAL_MS, load_config())
    assert decisions['AAAUSDT']['action'] == 'HOLD'
    assert decisions['AAAUSDT']['model_score'] == .8


def test_storage_pressure_drops_pending_samples_without_resetting_wallet_or_weights():
    import base64

    wallet = new_wallet(load_config())
    wallet['cash'] = 12.
    wallet['learning'] = new_learning_state(12)
    learning = wallet['learning']
    learning['market']['samples'] = 300
    learning['contexts'] = {'1': [0.] * CONTEXT_DIMENSIONS}
    learning['pending'] = [{'symbol': 'AAAUSDT', 'candle': i, 'technical': [.12345678] * 9,
                            'base': .12345678, 'shadow': .12345678, 'reference_price': 12345.12345678,
                            'context_id': '1'} for i in range(5000)]
    writes = []

    class GitHub:
        branch = 'test'

        def request(self, method, path, json):
            writes.append(json)

            class Response:
                status_code = 200

                def json(self):
                    return {'content': {'sha': 'next'}}

            return Response()

    store = WalletStore.__new__(WalletStore)
    store.github = GitHub()
    assert store.write(wallet, 'old') == 'next'
    payload = base64.b64decode(writes[0]['content'])
    assert len(payload) <= 800000
    persisted = json.loads(payload)
    assert persisted['cash'] == 12.
    assert persisted['learning']['market']['samples'] == 300
    assert persisted['learning_summary']['dropped_samples'] > 0
