import argparse
import base64
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import requests

from app.core import BASE_URL, FEATURES, INTERVAL, INTERVAL_MS, MODEL_DIR, connect, make_features

SYMBOLS = ('BTCUSDT', 'ETHUSDT')


def signal_connection():
    con = connect()
    con.executescript('''
        CREATE TABLE IF NOT EXISTS live_signals (
            symbol TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS signal_deliveries (
            symbol TEXT NOT NULL, candle_time INTEGER NOT NULL,
            status TEXT NOT NULL, PRIMARY KEY(symbol, candle_time));
    ''')
    return con


def save_snapshot(signal):
    with signal_connection() as con:
        con.execute('INSERT OR REPLACE INTO live_signals VALUES (?,?)',
                    (signal['symbol'], json.dumps(signal)))


def latest_signals():
    with signal_connection() as con:
        rows = con.execute('SELECT payload FROM live_signals ORDER BY symbol').fetchall()
    return [json.loads(row[0]) for row in rows]


class LocalDeliveryState:
    def reserve(self, signal):
        with signal_connection() as con:
            cursor = con.execute('INSERT OR IGNORE INTO signal_deliveries VALUES (?,?,?)',
                                 (signal['symbol'], signal['candle_time'], 'reserved'))
        return cursor.rowcount == 1

    def finish(self, signal, status):
        with signal_connection() as con:
            con.execute('UPDATE signal_deliveries SET status=? WHERE symbol=? AND candle_time=?',
                        (status, signal['symbol'], signal['candle_time']))


class GitHubDeliveryState:
    def __init__(self, branch='cryptoai-signal-state'):
        self.repository = os.environ.get('GITHUB_REPOSITORY', '')
        token = os.environ.get('GITHUB_TOKEN', '')
        if not self.repository or not token:
            raise ValueError('GITHUB_REPOSITORY and GITHUB_TOKEN are required.')
        self.branch = branch
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {token}',
                                     'Accept': 'application/vnd.github+json',
                                     'X-GitHub-Api-Version': '2022-11-28'})
        self.base = f'https://api.github.com/repos/{self.repository}'
        self.ensure_branch()

    def request(self, method, path, **kwargs):
        try:
            return self.session.request(method, self.base + path, timeout=25, **kwargs)
        except requests.RequestException:
            raise RuntimeError('GitHub state request failed; no Telegram retry will be attempted.') from None

    def ensure_branch(self):
        response = self.request('GET', f'/git/ref/heads/{self.branch}')
        if response.status_code == 200:
            return
        if response.status_code != 404:
            raise RuntimeError(f'Cannot read GitHub state branch ({response.status_code}).')
        repo = self.request('GET', '')
        if repo.status_code != 200:
            raise RuntimeError('Cannot read repository settings.')
        default = repo.json()['default_branch']
        ref = self.request('GET', f'/git/ref/heads/{default}')
        if ref.status_code != 200:
            raise RuntimeError('Cannot read default branch.')
        created = self.request('POST', '/git/refs', json={
            'ref': f'refs/heads/{self.branch}', 'sha': ref.json()['object']['sha']})
        if created.status_code != 201:
            exists = self.request('GET', f'/git/ref/heads/{self.branch}')
            if exists.status_code != 200:
                raise RuntimeError('Cannot create state branch; enable contents write permission.')

    def path(self, signal):
        return f"/contents/signal-state/{signal['symbol']}/{signal['candle_time']}.json"

    def reserve(self, signal):
        path = self.path(signal)
        existing = self.request('GET', path, params={'ref': self.branch})
        if existing.status_code == 200:
            return False
        if existing.status_code != 404:
            raise RuntimeError('Cannot check durable delivery state; sending is stopped.')
        payload = {**signal, 'delivery_status': 'reserved'}
        content = base64.b64encode(json.dumps(payload).encode()).decode()
        response = self.request('PUT', path, json={
            'message': f"Reserve {signal['symbol']} {signal['candle_time']}",
            'branch': self.branch, 'content': content})
        if response.status_code == 201:
            return True
        if response.status_code in (409, 422):
            existing = self.request('GET', path, params={'ref': self.branch})
            if existing.status_code == 200:
                return False
        raise RuntimeError('Cannot reserve delivery durably; Telegram was not called.')

    def finish(self, signal, status):
        path = self.path(signal)
        existing = self.request('GET', path, params={'ref': self.branch})
        if existing.status_code != 200:
            raise RuntimeError('Cannot update delivery status; reservation remains in place.')
        payload = {**signal, 'delivery_status': status}
        content = base64.b64encode(json.dumps(payload).encode()).decode()
        response = self.request('PUT', path, json={
            'message': f"Record delivery {signal['symbol']} {signal['candle_time']}",
            'branch': self.branch, 'sha': existing.json()['sha'], 'content': content})
        if response.status_code != 200:
            raise RuntimeError('Cannot update delivery status; reservation remains in place.')


def market_window(symbol, session=None):
    session = session or requests.Session()
    try:
        response = session.get(BASE_URL.replace('/klines', '/time'), timeout=25)
        response.raise_for_status()
        server_time = int(response.json()['serverTime'])
        end = server_time // INTERVAL_MS * INTERVAL_MS - 1
        response = session.get(BASE_URL, params={'symbol': symbol, 'interval': INTERVAL,
                                               'limit': 1000, 'endTime': end}, timeout=25)
        response.raise_for_status()
        rows = response.json()
    except requests.RequestException:
        raise RuntimeError(f'Binance request failed for {symbol}; no signal generated.') from None
    parsed = [{'open_time': int(row[0]), 'open': float(row[1]), 'high': float(row[2]),
               'low': float(row[3]), 'close': float(row[4]), 'volume': float(row[5])}
              for row in rows if int(row[6]) < server_time]
    frame = pd.DataFrame(parsed)
    expected = end + 1 - INTERVAL_MS
    if len(frame) < 300 or int(frame.iloc[-1]['open_time']) != expected:
        raise ValueError('Insufficient or stale closed candles; no signal generated.')
    if (frame['open_time'].diff().dropna() != INTERVAL_MS).any():
        raise ValueError('Candle intervals are missing or duplicated; no signal generated.')
    if not np.isfinite(frame.to_numpy()).all() or (frame[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError('Invalid market values; no signal generated.')
    return frame, server_time


def predict_signal(symbol, frame, bundle, server_time, model_id):
    if symbol not in SYMBOLS or bundle.get('symbol') != symbol or bundle.get('interval') != INTERVAL:
        raise ValueError('Model symbol/interval metadata is missing or mismatched; retrain models.')
    if bundle.get('features') != FEATURES:
        raise ValueError('Model feature schema does not match the live pipeline.')
    candle_time = int(frame.iloc[-1]['open_time'])
    if not isinstance(bundle.get('data_end_time'), int) or bundle['data_end_time'] >= candle_time:
        raise ValueError('Wait for a new closed candle after the model training data ends.')
    if candle_time != server_time // INTERVAL_MS * INTERVAL_MS - INTERVAL_MS:
        raise ValueError('Latest candle is stale; no signal generated.')
    if (frame['open_time'].diff().dropna() != INTERVAL_MS).any():
        raise ValueError('Missing candle intervals; no signal generated.')
    threshold = float(bundle.get('threshold', 0.65))
    horizon = int(bundle['horizon'])
    if not 0.5 <= threshold <= 1 or not 1 <= horizon <= 96:
        raise ValueError('Invalid model signal settings.')
    featured = make_features(frame)
    latest = featured.iloc[[-1]]
    if not np.isfinite(latest[FEATURES].to_numpy()).all():
        raise ValueError('Latest candle features are invalid; no signal generated.')
    probability = float(bundle['model'].predict_proba(latest[FEATURES])[0, 1])
    reference = float(latest.iloc[0]['close'])
    atr = float(latest.iloc[0]['atr'])
    if not np.isfinite(probability) or not 0 <= probability <= 1 or not np.isfinite(atr) or atr <= 0:
        raise ValueError('Invalid model output or ATR.')
    stop = reference - 1.5 * atr
    if stop <= 0:
        raise ValueError('Invalid reference stop.')
    return {'symbol': symbol, 'interval': INTERVAL, 'candle_time': candle_time,
            'candle_close_time': candle_time + INTERVAL_MS,
            'evaluated_at': datetime.now(timezone.utc).isoformat(),
            'action': 'BUY' if probability >= threshold else 'HOLD',
            'model_score': probability, 'threshold': threshold,
            'reference_price': reference, 'reference_stop': stop,
            'reference_target': reference + 3 * atr, 'horizon_bars': horizon,
            'model_id': model_id, 'model_trained_at': bundle.get('trained_at'),
            'delivery_status': 'not_requested'}


def telegram_message(signal):
    closed = datetime.fromtimestamp(signal['candle_close_time'] / 1000, timezone.utc)
    return (f"Crypto AI | {signal['symbol']} | Spot / 15m\n"
            f"سیگنال پژوهشی خرید — امتیاز مدل: {signal['model_score']:.3f}\n"
            f"بسته‌شدن کندل: {closed.isoformat()}\n"
            f"قیمت مرجع: {signal['reference_price']:.2f} USDT\n"
            f"حد ضرر مرجع: {signal['reference_stop']:.2f}\n"
            f"حد سود مرجع: {signal['reference_target']:.2f}\n"
            f"افق: {signal['horizon_bars'] * 15} دقیقه\n"
            'قیمت‌ها از کلوز کندل‌اند، نه قیمت اجرای سفارش. '
            'امتیاز، احتمال کالیبره‌شده نیست؛ سفارش واقعی اجرا نمی‌شود.')


def send_telegram(signal):
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
    if not token or not chat_id:
        raise ValueError('Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before sending.')
    try:
        response = requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                                 json={'chat_id': chat_id, 'text': telegram_message(signal)},
                                 timeout=25)
        payload = response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError('Telegram delivery outcome is unknown; automatic retry is disabled.') from None
    if not response.ok or not payload.get('ok'):
        raise RuntimeError('Telegram rejected delivery; automatic retry is disabled.')
    return payload['result']['message_id']


def deliver(signal, delivery_state, sender=send_telegram):
    if signal['action'] != 'BUY':
        signal['delivery_status'] = 'not_required'
        save_snapshot(signal)
        return
    if not delivery_state.reserve(signal):
        signal['delivery_status'] = 'duplicate_suppressed'
        save_snapshot(signal)
        return
    try:
        sender(signal)
    except Exception:
        signal['delivery_status'] = 'unknown_or_failed'
        save_snapshot(signal)
        delivery_state.finish(signal, 'unknown_or_failed')
        raise
    signal['delivery_status'] = 'sent'
    save_snapshot(signal)
    delivery_state.finish(signal, 'sent')


def run_signals(send=False, state_backend='local'):
    if send and not all(os.environ.get(key) for key in ('TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID')):
        raise ValueError('Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before sending.')
    delivery_state = (GitHubDeliveryState() if state_backend == 'github' else LocalDeliveryState()) if send else None
    results, failures = [], []
    for symbol in SYMBOLS:
        try:
            path = MODEL_DIR / f'{symbol}_xgboost.joblib'
            bundle = joblib.load(path)
            frame, server_time = market_window(symbol)
            signal = predict_signal(symbol, frame, bundle, server_time,
                                    hashlib.sha256(path.read_bytes()).hexdigest()[:16])
            if send:
                deliver(signal, delivery_state)
            else:
                signal['delivery_status'] = 'preview'
                save_snapshot(signal)
            results.append(signal)
        except Exception as error:
            message = str(error) if isinstance(error, (ValueError, RuntimeError, FileNotFoundError)) else type(error).__name__
            failed = {'symbol': symbol, 'action': 'ERROR', 'error': message,
                      'evaluated_at': datetime.now(timezone.utc).isoformat(), 'delivery_status': 'error'}
            save_snapshot(failed)
            failures.append(failed)
    print(json.dumps({'signals': results, 'errors': failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description='Evaluate latest closed candles using saved models.')
    parser.add_argument('--send', action='store_true')
    parser.add_argument('--state-backend', choices=['local', 'github'], default='local')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval-seconds', type=int, default=600)
    args = parser.parse_args()
    if args.interval_seconds < 60:
        parser.error('--interval-seconds must be at least 60')
    try:
        while True:
            result = run_signals(args.send, args.state_backend)
            if not args.watch:
                raise SystemExit(result)
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        parser.exit(0, 'Signal watcher stopped.\n')
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
