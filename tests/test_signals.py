from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import requests

from app import core, signals
from tests.test_pipeline import synthetic


class FixedModel:
    def __init__(self, score):
        self.score = score
        self.seen_time = None

    def predict_proba(self, frame):
        self.seen_time = len(frame)
        return np.array([[1 - self.score, self.score]])


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'DATA', tmp_path)
    monkeypatch.setattr(core, 'DB', tmp_path / 'test.db')


@pytest.fixture
def candidate():
    frame = synthetic(1000)
    bundle = {'model': FixedModel(0.7), 'features': core.FEATURES,
              'horizon': 12, 'symbol': 'BTCUSDT', 'interval': '15m',
              'threshold': 0.65, 'data_end_time': int(frame.iloc[-2]['open_time'])}
    now = int(frame.iloc[-1]['open_time']) + core.INTERVAL_MS + 1000
    return frame, bundle, now


def make_signal(candidate):
    frame, bundle, now = candidate
    return signals.predict_signal('BTCUSDT', frame, bundle, now, 'test-model')


def test_latest_closed_candle_is_scored_without_future_labels(candidate):
    signal = make_signal(candidate)
    assert signal['action'] == 'BUY'
    assert signal['candle_time'] == int(candidate[0].iloc[-1]['open_time'])
    assert candidate[1]['model'].seen_time == 1
    assert signal['reference_stop'] < signal['reference_price'] < signal['reference_target']


def test_scores_below_threshold_do_not_trigger_buy(candidate):
    candidate[1]['model'] = FixedModel(0.64)
    assert make_signal(candidate)['action'] == 'HOLD'


def test_freshly_trained_models_wait_without_scoring_or_sending(candidate, tmp_path, monkeypatch, capsys):
    frame, bundle, now = candidate
    bundle['data_end_time'] = int(frame.iloc[-1]['open_time'])
    (tmp_path / 'BTCUSDT_xgboost.joblib').write_bytes(b'test-model')
    snapshots, sent = [], []
    monkeypatch.setattr(signals, 'MODEL_DIR', tmp_path)
    monkeypatch.setattr(signals, 'SYMBOLS', ('BTCUSDT',))
    monkeypatch.setattr(signals.joblib, 'load', lambda _: bundle)
    monkeypatch.setattr(signals, 'market_window', lambda _: (frame, now))
    monkeypatch.setattr(signals, 'save_snapshot', snapshots.append)
    monkeypatch.setattr(signals, 'LocalDeliveryState', lambda: None)
    monkeypatch.setattr(signals, 'deliver', lambda signal, _: sent.append(signal))
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test-token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', 'test-chat')
    assert signals.run_signals(send=True) == 0
    assert snapshots[0]['action'] == 'WAITING'
    assert snapshots[0]['delivery_status'] == 'waiting_for_new_candle'
    assert bundle['model'].seen_time is None
    assert not sent
    assert '"errors": []' in capsys.readouterr().out


@pytest.mark.parametrize('corruption', ['stale', 'gap', 'trained_on_latest', 'wrong_symbol', 'invalid_score'])
def test_unsafe_inputs_cannot_generate_a_signal(candidate, corruption):
    frame, bundle, now = candidate
    if corruption == 'stale':
        now += core.INTERVAL_MS
    elif corruption == 'gap':
        frame = frame.drop(index=300)
    elif corruption == 'trained_on_latest':
        bundle['data_end_time'] = int(frame.iloc[-1]['open_time'])
    elif corruption == 'wrong_symbol':
        bundle['symbol'] = 'ETHUSDT'
    else:
        bundle['model'] = FixedModel(float('nan'))
    with pytest.raises(ValueError):
        signals.predict_signal('BTCUSDT', frame, bundle, now, 'test-model')


def test_restarts_and_model_changes_do_not_resend_same_candle(candidate, isolated_db):
    signal = make_signal(candidate)
    sent = []
    signals.deliver(deepcopy(signal), signals.LocalDeliveryState(), sent.append)
    signal['model_id'] = 'retrained-model'
    signals.deliver(signal, signals.LocalDeliveryState(), sent.append)
    assert len(sent) == 1
    assert signal['delivery_status'] == 'duplicate_suppressed'


def test_concurrent_workers_can_reserve_a_candle_only_once(candidate, isolated_db):
    signal = make_signal(candidate)
    with signals.signal_connection():
        pass
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: signals.LocalDeliveryState().reserve(signal), range(2)))
    assert sorted(results) == [False, True]


def test_uncertain_delivery_is_never_retried(candidate, isolated_db):
    signal = make_signal(candidate)

    def uncertain(_):
        raise RuntimeError('Timeout after possible acceptance')

    with pytest.raises(RuntimeError):
        signals.deliver(signal, signals.LocalDeliveryState(), uncertain)
    sent = []
    signals.deliver(deepcopy(signal), signals.LocalDeliveryState(), sent.append)
    assert not sent


def test_hold_does_not_reserve_or_send(candidate, isolated_db):
    signal = make_signal(candidate)
    signal['action'] = 'HOLD'
    sent = []
    signals.deliver(signal, signals.LocalDeliveryState(), sent.append)
    assert not sent
    assert signals.LocalDeliveryState().reserve(signal)


def test_new_candle_and_other_symbol_have_distinct_delivery_keys(candidate, isolated_db):
    signal = make_signal(candidate)
    state = signals.LocalDeliveryState()
    assert state.reserve(signal)
    assert state.reserve({**signal, 'symbol': 'ETHUSDT'})
    assert state.reserve({**signal, 'candle_time': signal['candle_time'] + core.INTERVAL_MS})


def test_state_failure_prevents_telegram_call(candidate, isolated_db):
    class BrokenState:
        def reserve(self, _):
            raise RuntimeError('State unavailable')

    sent = []
    with pytest.raises(RuntimeError):
        signals.deliver(make_signal(candidate), BrokenState(), sent.append)
    assert not sent


class Response:
    def __init__(self, status, payload=None):
        self.status_code = status
        self.payload = payload or {}

    def json(self):
        return self.payload


def test_github_existing_reservation_suppresses_retries(candidate):
    state = object.__new__(signals.GitHubDeliveryState)
    state.branch = 'cryptoai-signal-state'
    state.request = lambda *args, **kwargs: Response(200)
    assert not state.reserve(make_signal(candidate))


def test_github_reserves_before_send_and_refuses_unconfirmed_write(candidate):
    state = object.__new__(signals.GitHubDeliveryState)
    state.branch = 'cryptoai-signal-state'
    responses = iter([Response(404), Response(500)])
    state.request = lambda *args, **kwargs: next(responses)
    with pytest.raises(RuntimeError):
        state.reserve(make_signal(candidate))


def test_github_create_conflict_checks_for_existing_reservation(candidate):
    state = object.__new__(signals.GitHubDeliveryState)
    state.branch = 'cryptoai-signal-state'
    responses = iter([Response(404), Response(422), Response(200)])
    state.request = lambda *args, **kwargs: next(responses)
    assert not state.reserve(make_signal(candidate))


def test_telegram_network_error_does_not_leak_bot_token(candidate, monkeypatch):
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'test-secret-token')
    monkeypatch.setenv('TELEGRAM_CHAT_ID', '123')

    def failed(*args, **kwargs):
        raise requests.ConnectionError('https://api.telegram.org/bottest-secret-token/sendMessage')

    monkeypatch.setattr(signals.requests, 'post', failed)
    with pytest.raises(RuntimeError) as error:
        signals.send_telegram(make_signal(candidate))
    assert 'test-secret-token' not in str(error.value)
