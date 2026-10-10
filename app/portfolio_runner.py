import base64
import json
import os
from datetime import datetime, timezone

import joblib
import requests

from app.core import MODEL_DIR
from app.market import book_quotes, candle_windows, public_data
from app.portfolio import load_config, new_wallet, paper_cycle, validate_wallet
from app.portfolio_models import predict
from app.signals import GitHubDeliveryState


class WalletStore:
    def __init__(self):
        self.github = GitHubDeliveryState(branch='cryptoai-portfolio')

    def read(self, config):
        response = self.github.request('GET', '/contents/paper/portfolio.json', params={'ref': self.github.branch})
        if response.status_code == 404:
            return new_wallet(config), None
        if response.status_code != 200:
            raise RuntimeError('Cannot read durable wallet; refusing to reset capital.')
        payload = response.json()
        state = json.loads(base64.b64decode(payload['content']))
        validate_wallet(state)
        return state, payload['sha']

    def write(self, state, sha):
        content = json.dumps(state, ensure_ascii=False, allow_nan=False).encode()
        if len(content) > 800_000:
            raise ValueError('Wallet exceeds storage budget.')
        body = {'branch': self.github.branch, 'message': 'Update shared paper wallet',
                'content': base64.b64encode(content).decode()}
        if sha:
            body['sha'] = sha
        response = self.github.request('PUT', '/contents/paper/portfolio.json', json=body)
        if response.status_code not in (200, 201):
            raise RuntimeError('Wallet conflict or persistence failure; no report sent.')
        return response.json()['content']['sha']


def report_text(state):
    timestamp = datetime.fromtimestamp(state['updated_at'] / 1000, timezone.utc).isoformat()
    positions = ', '.join(state['positions']) or 'ندارد'
    events = '\n'.join(f"{event['side']} {event['symbol']} | {event['notional']:.2f} USDT | {event['reason']}"
                       for event in state['last_events']) or 'در این نوبت معامله‌ای انجام نشد.'
    return (f"Cryptoai | کیف پول مجازی مشترک | {len(state['universe'])} ارز\n"
            f"شروع: {state['initial_balance']:.2f} USDT\n"
            f"ارزش خالص: {state['equity']:.4f} USDT\n"
            f"نقد: {state['cash']:.4f} USDT\n"
            f"سود/زیان کل: {state['total_pnl']:+.4f} USDT\n"
            f"سود/زیان بسته‌شده: {state['realized_pnl']:+.4f} USDT\n"
            f"کارمزد: {state['fees_paid']:.4f} USDT | افت: {state['max_drawdown']:.2%}\n"
            f"موقعیت‌های باز: {positions}\n{events}\n"
            f"خطای بازار: {len(state['market_errors'])} | توقف خرید: {state['halted']}\n"
            f"زمان UTC: {timestamp}\n"
            'معاملات شبیه‌سازی هستند؛ پول واقعی جابه‌جا نمی‌شود.')


def send_report(state):
    token, chat = os.environ.get('TELEGRAM_BOT_TOKEN'), os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat:
        raise ValueError('Telegram repository secrets are missing.')
    try:
        response = requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                                 json={'chat_id': chat, 'text': report_text(state)}, timeout=25)
        if not response.ok or not response.json().get('ok'):
            raise RuntimeError('Telegram rejected report.')
    except (requests.RequestException, ValueError):
        raise RuntimeError('Telegram report outcome unknown; no automatic retry.') from None


def persist_and_report(store, state, sha, config, sender=send_report):
    report_slot = state['updated_at'] // (config['report_interval_minutes'] * 60000)
    due = state['last_report_slot'] is None or report_slot > state['last_report_slot']
    if due:
        state['last_report_slot'] = report_slot
        state['report_status'] = 'reserved'
    sha = store.write(state, sha)
    if not due:
        return
    try:
        sender(state)
    except Exception:
        state['report_status'] = 'unknown_or_failed'
        store.write(state, sha)
        raise
    state['report_status'] = 'sent'
    store.write(state, sha)


def run():
    config = load_config()
    if not all(os.environ.get(key) for key in ('TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID')):
        raise ValueError('Telegram repository secrets are missing.')
    bundle = joblib.load(MODEL_DIR / 'portfolio.joblib')
    markets = {item['symbol']: item for item in bundle['markets']}
    if len(markets) != config['universe_size']:
        raise ValueError('Incomplete portfolio universe.')
    store = WalletStore()
    wallet, sha = store.read(config)
    now = int(public_data('/time')['serverTime'])
    if wallet['last_cycle_slot'] is not None and now // 600000 <= wallet['last_cycle_slot']:
        print('Cycle already persisted; duplicate suppressed.')
        return
    frames, errors = candle_windows(markets, 300, now)
    signals = {}
    for symbol, frame in frames.items():
        try:
            signals[symbol] = predict(symbol, frame, bundle, now, config)
        except ValueError as error:
            errors[symbol] = str(error)
    quotes = book_quotes(set(markets) | set(wallet['positions']))
    execution_time = int(public_data('/time')['serverTime'])
    if execution_time - now > 120000:
        raise RuntimeError('Market collection took too long; no stale execution.')
    for symbol in set(markets) | set(wallet['positions']):
        if symbol not in quotes:
            errors[symbol] = 'Execution quote unavailable.'
    state, events = paper_cycle(wallet, signals, quotes, markets, errors, execution_time, config)
    state['universe'] = list(markets)
    state['model_metrics'] = bundle['metrics']
    state['model_trained_at'] = bundle['trained_at']
    persist_and_report(store, state, sha, config)
    print(json.dumps({'equity': state['equity'], 'cash': state['cash'], 'events': events,
                      'errors': errors, 'report': state['report_status']}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        run()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
