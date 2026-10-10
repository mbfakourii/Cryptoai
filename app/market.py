import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from app.core import BASE_URL, INTERVAL, INTERVAL_MS

API = BASE_URL.rsplit('/', 1)[0]
STABLE_ASSETS = frozenset(('USDT', 'USDC', 'FDUSD', 'TUSD', 'USDP', 'DAI', 'USDE',
                         'USD1', 'BUSD', 'EURI', 'AEUR', 'XUSD', 'USDD', 'PYUSD', 'RLUSD'))


def public_data(path, params=None):
    try:
        response = requests.get(API + path, params=params, timeout=25)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        raise RuntimeError(f'Public market data unavailable: {path}') from None


def select_universe(exchange_info, tickers, size=50):
    volumes = {item['symbol']: float(item.get('quoteVolume', 0)) for item in tickers}
    eligible = []
    for item in exchange_info['symbols']:
        if (item['quoteAsset'] != 'USDT' or item['status'] != 'TRADING'
                or not item.get('isSpotTradingAllowed') or item['baseAsset'] in STABLE_ASSETS):
            continue
        filters = {row['filterType']: row for row in item['filters']}
        lot = filters.get('LOT_SIZE')
        minimum = filters.get('NOTIONAL', filters.get('MIN_NOTIONAL'))
        volume = volumes.get(item['symbol'], 0)
        if not lot or not minimum or not math.isfinite(volume) or volume <= 0:
            continue
        eligible.append({'symbol': item['symbol'], 'asset': item['baseAsset'],
                         'quote_volume': volume, 'step_size': lot['stepSize'],
                         'min_qty': lot['minQty'], 'max_qty': lot['maxQty'],
                         'min_notional': float(minimum['minNotional'])})
    eligible.sort(key=lambda item: (-item['quote_volume'], item['symbol']))
    if len(eligible) < size:
        raise ValueError('Insufficient eligible USDT spot markets.')
    return eligible[:size]


def universe(size=50):
    return select_universe(public_data('/exchangeInfo'), public_data('/ticker/24hr'), size)


def candle_frame(symbol, count, server_time):
    end = server_time // INTERVAL_MS * INTERVAL_MS - 1
    rows = []
    while len(rows) < count:
        chunk = public_data('/klines', {'symbol': symbol, 'interval': INTERVAL,
                                       'limit': min(1000, count - len(rows)), 'endTime': end})
        if not chunk:
            break
        rows = chunk + rows
        end = int(chunk[0][0]) - 1
        if len(rows) < count:
            time.sleep(0.12)
    parsed = [{'open_time': int(row[0]), 'open': float(row[1]), 'high': float(row[2]),
               'low': float(row[3]), 'close': float(row[4]), 'volume': float(row[5])}
              for row in rows if int(row[6]) < server_time]
    frame = pd.DataFrame(parsed)
    expected = server_time // INTERVAL_MS * INTERVAL_MS - INTERVAL_MS
    if len(frame) != count or int(frame.iloc[-1]['open_time']) != expected:
        raise ValueError(f'Insufficient or stale candles for {symbol}.')
    if (frame['open_time'].diff().dropna() != INTERVAL_MS).any():
        raise ValueError(f'Missing candle intervals for {symbol}.')
    if (not all(math.isfinite(value) for value in frame.to_numpy().ravel())
            or (frame[['open', 'high', 'low', 'close']] <= 0).any().any()
            or (frame['volume'] < 0).any()):
        raise ValueError(f'Invalid market values for {symbol}.')
    return frame


def candle_windows(symbols, count, server_time):
    frames, errors = {}, {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        tasks = {pool.submit(candle_frame, symbol, count, server_time): symbol for symbol in symbols}
        for task in as_completed(tasks):
            symbol = tasks[task]
            try:
                frames[symbol] = task.result()
            except (RuntimeError, ValueError) as error:
                errors[symbol] = str(error)
    return frames, errors


def book_quotes(symbols):
    wanted = set(symbols)
    quotes = {}
    for item in public_data('/ticker/bookTicker'):
        if item['symbol'] not in wanted:
            continue
        bid, ask = float(item['bidPrice']), float(item['askPrice'])
        if math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask:
            quotes[item['symbol']] = {'bid': bid, 'ask': ask}
    return quotes
