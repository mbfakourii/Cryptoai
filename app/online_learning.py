import copy
import math

import numpy as np
from sklearn.linear_model import SGDClassifier

from app.core import FEATURES, INTERVAL_MS, make_features
from app.market_context import CONTEXT_DIMENSIONS

DIMENSIONS = 1 + len(FEATURES) + CONTEXT_DIMENSIONS
MAX_PENDING = 2000


def new_classifier(base_weight=1.):
    weights = [0.] * DIMENSIONS
    weights[0] = base_weight
    return {'weights': weights, 'intercept': 0., 'steps': 1., 'samples': 0,
            'evaluated': 0, 'brier': None, 'baseline_brier': None}


def new_learning_state(horizon):
    return {'schema': 1, 'horizon': horizon, 'market': new_classifier(),
            'feedback': new_classifier(0.), 'pending': [], 'contexts': {},
            'entries': {}, 'last_candles': {}, 'last_update': None, 'dropped_samples': 0}


def validate_learning(state, horizon):
    if state.get('schema') != 1 or state.get('horizon') != horizon:
        raise ValueError('Online learning schema or horizon changed; migrate the stored learner explicitly.')
    for name in ('market', 'feedback'):
        model = state[name]
        if (len(model['weights']) != DIMENSIONS
                or not all(math.isfinite(value) for value in model['weights'] + [model['intercept']])
                or model['samples'] < 0):
            raise ValueError('Invalid stored online model.')


def probability(model, vector):
    logit = float(np.dot(model['weights'], vector) + model['intercept'])
    return float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))


def learn(model, vectors, targets):
    if not vectors:
        return
    estimator = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=.01,
                              alpha=.001, shuffle=False, random_state=42)
    estimator.partial_fit(np.zeros((1, DIMENSIONS)), [0], classes=[0, 1])
    estimator.coef_ = np.asarray([model['weights']], dtype=float)
    estimator.intercept_ = np.asarray([model['intercept']], dtype=float)
    estimator.t_ = model['steps']
    estimator.partial_fit(np.asarray(vectors), np.asarray(targets))
    model.update(weights=estimator.coef_[0].tolist(), intercept=float(estimator.intercept_[0]),
                 steps=float(estimator.t_), samples=model['samples'] + len(targets))


def evaluate(model, score, baseline, target):
    for key, prediction in (('brier', score), ('baseline_brier', baseline)):
        loss = (prediction - target) ** 2
        model[key] = loss if model[key] is None else .99 * model[key] + .01 * loss
    model['evaluated'] += 1


def active(model, minimum):
    return (model['samples'] >= minimum and model['evaluated'] >= minimum
            and model['brier'] is not None and model['brier'] < model['baseline_brier'])


def feature_vector(technical, context, base_score):
    clipped = min(max(base_score, .001), .999)
    scales = [100, 100, 100, 1, 100, .01, 100, 100, 100]
    values = [math.log(clipped / (1 - clipped))]
    values.extend(max(-5., min(5., float(value) * scale)) for value, scale in zip(technical, scales))
    values.extend(context)
    if len(values) != DIMENSIONS or not all(math.isfinite(value) for value in values):
        raise ValueError('Invalid online features.')
    return [round(value, 6) for value in values]


def mature_samples(state, frames, now):
    remaining, vectors, targets = [], [], []
    indexed = {symbol: frame.set_index('open_time') for symbol, frame in frames.items()}
    for sample in state['pending']:
        end = sample['candle'] + state['horizon'] * INTERVAL_MS
        if end + INTERVAL_MS > now:
            remaining.append(sample)
            continue
        frame = indexed.get(sample['symbol'])
        if frame is None or end not in frame.index:
            if now - end < 2 * 86400000:
                remaining.append(sample)
            else:
                state['dropped_samples'] += 1
            continue
        target = int(float(frame.loc[end, 'close']) / sample['reference_price'] - 1 > .004)
        vector = feature_vector(sample['technical'], state['contexts'][sample['context_id']], sample['base'])
        evaluate(state['market'], sample['shadow'], sample['base'], target)
        vectors.append(vector)
        targets.append(target)
    learn(state['market'], vectors, targets)
    state['pending'] = remaining
    if targets:
        state['last_update'] = now


def adapt_signals(stored, frames, signals, quotes, context, now, config):
    state = copy.deepcopy(stored) if stored is not None else new_learning_state(config['horizon_bars'])
    if state.get('schema') == 1 and state.get('horizon') == 12 and config['horizon_bars'] == 4:
        previous = state
        state = new_learning_state(4)
        state['migration'] = {'from_horizon': 12, 'to_horizon': 4, 'at': now,
                              'previous_market_samples': previous['market']['samples'],
                              'previous_trade_samples': previous['feedback']['samples'],
                              'discarded_pending': len(previous['pending'])}
        state['last_candles'] = dict(previous['last_candles'])
        state['dropped_samples'] = previous['dropped_samples'] + len(previous['pending'])
    validate_learning(state, config['horizon_bars'])
    mature_samples(state, frames, now)
    decisions = copy.deepcopy(signals)
    snapshots = {}
    for symbol, signal in decisions.items():
        if signal['action'] not in ('BUY', 'HOLD') or symbol not in quotes:
            continue
        candle = signal['candle_time']
        if candle <= state['last_candles'].get(symbol, -1):
            signal['action'] = 'WAITING'
            continue
        latest = make_features(frames[symbol]).iloc[-1]
        technical = [round(float(latest[key]), 8) for key in FEATURES]
        base = signal['model_score']
        vector = feature_vector(technical, context['vector'], base)
        shadow = .5 * base + .5 * probability(state['market'], vector)
        feedback = probability(state['feedback'], vector)
        signal['base_model_score'] = base
        if context['ready'] and active(state['market'], 200):
            signal['model_score'] = shadow
        signal['action'] = 'BUY' if signal['model_score'] >= config['entry_threshold'] else 'HOLD'
        if (not context['ready'] or not signal.get('entry_allowed', True)
                or (active(state['feedback'], 30) and feedback < .5)):
            signal['action'] = 'HOLD'
        state['pending'].append({'symbol': symbol, 'candle': candle, 'technical': technical,
                                 'base': base, 'shadow': shadow, 'reference_price': quotes[symbol]['ask'],
                                 'context_id': str(now)})
        state['contexts'][str(now)] = context['vector']
        state['last_candles'][symbol] = candle
        snapshots[symbol] = {'vector': vector, 'feedback_score': feedback}
    if len(state['pending']) > MAX_PENDING:
        trim_pending(state, len(state['pending']) - MAX_PENDING)
    used = {item['context_id'] for item in state['pending']}
    state['contexts'] = {key: value for key, value in state['contexts'].items() if key in used}
    return state, decisions, snapshots


def record_trade_outcomes(state, events, snapshots):
    for event in events:
        symbol = event['symbol']
        if event['side'] == 'BUY' and symbol in snapshots:
            state['entries'][symbol] = {**snapshots[symbol], 'entry_time': event['time']}
        elif event['side'] == 'SELL':
            entry = state['entries'].get(symbol)
            if entry and entry['entry_time'] == event['entry_time']:
                target = int(event['pnl'] > 0)
                evaluate(state['feedback'], entry['feedback_score'], .5, target)
                learn(state['feedback'], [entry['vector']], [target])
                del state['entries'][symbol]
                state['last_update'] = event['time']


def learning_summary(state):
    return {'market_samples': state['market']['samples'],
            'closed_trade_samples': state['feedback']['samples'],
            'market_active': active(state['market'], 200),
            'feedback_active': active(state['feedback'], 30),
            'last_update': state['last_update'], 'pending_samples': len(state['pending']),
            'dropped_samples': state['dropped_samples'],
            'horizon_bars': state['horizon'], 'migration': state.get('migration'),
            'brier': state['market']['brier'], 'baseline_brier': state['market']['baseline_brier']}


def trim_pending(state, count):
    removed = min(count, len(state['pending']))
    state['pending'] = state['pending'][removed:]
    state['dropped_samples'] += removed
    used = {item['context_id'] for item in state['pending']}
    state['contexts'] = {key: value for key, value in state['contexts'].items() if key in used}
