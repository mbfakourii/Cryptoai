"""Data acquisition, leakage-aware features, model fitting and spot-only backtesting."""
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
DB = DATA / 'crypto.db'
MODEL_DIR = DATA / 'models'
BASE_URL = 'https://data-api.binance.vision/api/v3/klines'
INTERVAL = '15m'
INTERVAL_MS = 15 * 60 * 1000
FEATURES = ['return_1', 'return_4', 'return_12', 'volume_ratio', 'ema_gap',
            'rsi_14', 'atr_pct', 'range_pct', 'volatility_20']


def connect():
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute('PRAGMA journal_mode=WAL')
    con.executescript('''
    CREATE TABLE IF NOT EXISTS candles (
      symbol TEXT NOT NULL, open_time INTEGER NOT NULL, open REAL NOT NULL,
      high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
      volume REAL NOT NULL, PRIMARY KEY(symbol, open_time));
    CREATE TABLE IF NOT EXISTS model_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
      created_at TEXT NOT NULL, train_rows INTEGER NOT NULL,
      validation_rows INTEGER NOT NULL, test_rows INTEGER NOT NULL,
      metrics_json TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS trades (
      id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
      symbol TEXT NOT NULL, signal_time INTEGER NOT NULL,
      entry_time INTEGER NOT NULL, exit_time INTEGER NOT NULL,
      entry_price REAL NOT NULL, exit_price REAL NOT NULL,
      probability REAL NOT NULL, result TEXT NOT NULL,
      net_return REAL NOT NULL, FOREIGN KEY(run_id) REFERENCES model_runs(id));
    ''')
    return con


def fetch_candles(symbol='BTCUSDT', limit=5000):
    if symbol not in ('BTCUSDT', 'ETHUSDT'):
        raise ValueError('Supported symbols: BTCUSDT, ETHUSDT')
    if not 300 <= limit <= 50000:
        raise ValueError('limit must be 300..50000')
    now_ms = int(time.time() * 1000)
    # Query backwards using endTime to obtain historical closed candles.
    end = (now_ms // INTERVAL_MS) * INTERVAL_MS - 1
    rows = []
    session = requests.Session()
    while len(rows) < limit:
        batch = min(1000, limit - len(rows))
        response = session.get(BASE_URL, params={'symbol': symbol, 'interval': INTERVAL,
                                                 'limit': batch, 'endTime': end}, timeout=25)
        response.raise_for_status()
        chunk = response.json()
        if not chunk:
            break
        rows = chunk + rows
        end = int(chunk[0][0]) - 1
        time.sleep(0.12)
    parsed = [(symbol, int(r[0]), *[float(r[j]) for j in (1, 2, 3, 4, 5)])
              for r in rows if int(r[6]) < now_ms]
    with connect() as con:
        con.executemany('INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?)', parsed)
    return len(parsed)


def load_candles(symbol):
    with connect() as con:
        df = pd.read_sql_query('SELECT * FROM candles WHERE symbol=? ORDER BY open_time',
                               con, params=(symbol,))
    if df.empty:
        raise ValueError('No data. Run fetch first.')
    return df


def make_features(df):
    df = df.sort_values('open_time').drop_duplicates('open_time').copy().reset_index(drop=True)
    close, high, low, volume = (df[c] for c in ('close', 'high', 'low', 'volume'))
    df['return_1'] = close.pct_change(1)
    df['return_4'] = close.pct_change(4)
    df['return_12'] = close.pct_change(12)
    df['volume_ratio'] = volume / volume.rolling(20).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    df['ema_gap'] = (ema20 - ema50) / close
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['atr_pct'] = df['atr'] / close
    df['range_pct'] = (high-low) / close
    df['volatility_20'] = df['return_1'].rolling(20).std()
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def label_data(df, horizon=12, threshold=0.002):
    """Label uses next bar OPEN as entry, and close of horizon's last bar as exit.
    Signal at row i, entry i+1, horizon bars: i+1 through i+horizon.
    """
    out = df.copy()
    out['future_entry'] = out['open'].shift(-1)
    out['future_exit'] = out['close'].shift(-horizon)
    out['target'] = ((out['future_exit'] / out['future_entry'] - 1) > threshold).astype(float)
    out.loc[out['future_exit'].isna() | out['future_entry'].isna(), 'target'] = np.nan
    return out.dropna(subset=FEATURES + ['target']).reset_index(drop=True)


def split_time(df, horizon=12):
    """Purged train/validation/test; prevents labels crossing the next split."""
    n = len(df)
    a, b = int(n * 0.60), int(n * 0.80)
    train = df.iloc[:max(0, a-horizon)]
    val = df.iloc[a:max(a, b-horizon)]
    test = df.iloc[b:]
    if min(len(train), len(val), len(test)) < 80:
        raise ValueError('Not enough data for purged time splits (fetch more candles).')
    return train, val, test


def simulate(df, probs, threshold=0.65, horizon=12, fee_bps=10, sl_atr=1.5, tp_atr=3.0):
    """Non-overlapping long-only spot trades; stop-first if both hit same bar.
    Signal at close of i; next candle open is entry; no look-ahead for signals.
    """
    trades = []
    next_allowed = 0
    fee = fee_bps / 10000
    for i, prob in enumerate(probs):
        if i < next_allowed or prob < threshold or i + horizon >= len(df):
            continue
        entry_idx = i + 1
        entry = float(df.iloc[entry_idx]['open'])
        atr = float(df.iloc[i]['atr'])
        if not np.isfinite(atr) or atr <= 0:
            continue
        stop, take = entry - sl_atr * atr, entry + tp_atr * atr
        if stop <= 0:
            continue
        exit_idx, exit_price, reason = i + horizon, float(df.iloc[i+horizon]['close']), 'TIME'
        for j in range(entry_idx, i + horizon + 1):
            bar = df.iloc[j]
            # Conservative assumption for candles crossing both levels.
            if float(bar['low']) <= stop:
                exit_idx, exit_price, reason = j, stop, 'SL'
                break
            if float(bar['high']) >= take:
                exit_idx, exit_price, reason = j, take, 'TP'
                break
        # Round-trip taker fee and a conservative slippage allowance (both sides).
        net_return = (exit_price / entry - 1) - 2 * fee - 2 * 0.0002
        trades.append({'signal_time': int(df.iloc[i]['open_time']),
                       'entry_time': int(df.iloc[entry_idx]['open_time']),
                       'exit_time': int(df.iloc[exit_idx]['open_time']),
                       'entry_price': entry, 'exit_price': exit_price,
                       'probability': float(prob), 'result': reason,
                       'net_return': float(net_return)})
        next_allowed = exit_idx + 1
    return trades


def train_and_backtest(symbol='BTCUSDT', horizon=12, probability_threshold=0.65):
    raw = load_candles(symbol)
    if (raw['open_time'].diff().dropna() != INTERVAL_MS).any():
        raise ValueError('Missing candle intervals detected; refill data before training.')
    featured = make_features(raw)
    labelled = label_data(featured, horizon=horizon)
    train, val, test = split_time(labelled, horizon)
    model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04,
                          subsample=0.8, colsample_bytree=0.9, objective='binary:logistic',
                          eval_metric='logloss', random_state=42, n_jobs=2)
    model.fit(train[FEATURES], train['target'].astype(int),
              eval_set=[(val[FEATURES], val['target'].astype(int))], verbose=False)
    val_probs = model.predict_proba(val[FEATURES])[:, 1]
    test_probs = model.predict_proba(test[FEATURES])[:, 1]
    y = test['target'].astype(int)
    trades = simulate(test, test_probs, threshold=probability_threshold, horizon=horizon)
    returns = np.array([t['net_return'] for t in trades], dtype=float)
    equity = np.cumprod(1 + returns) if len(returns) else np.array([])
    peak = np.maximum.accumulate(np.r_[1., equity]) if len(returns) else np.array([1.])
    curve = np.r_[1., equity] if len(returns) else np.array([1.])
    metrics = {
        'symbol': symbol, 'interval': INTERVAL, 'horizon_bars': horizon,
        'signal_probability_threshold': probability_threshold,
        'train_rows': len(train), 'validation_rows': len(val), 'test_rows': len(test),
        'test_accuracy': float(accuracy_score(y, test_probs >= 0.5)),
        'test_brier_score': float(brier_score_loss(y, test_probs)),
        'test_auc': float(roc_auc_score(y, test_probs)) if y.nunique() == 2 else None,
        'validation_brier_score': float(brier_score_loss(val['target'].astype(int), val_probs)),
        'trade_count': len(trades),
        'win_rate': float(np.mean(returns > 0)) if len(returns) else None,
        'total_return_compounded': float(equity[-1]-1) if len(returns) else 0.,
        'max_drawdown': float(np.min(curve/peak-1)),
        'profit_factor': (float(returns[returns > 0].sum() / -returns[returns < 0].sum())
                          if (returns < 0).any() else None),
        'fees_bps_per_side': 10, 'slippage_bps_per_side': 2,
        'note': 'Research backtest only. No live trading; model probabilities not calibrated.'
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES, 'horizon': horizon,
                 'symbol': symbol, 'interval': INTERVAL,
                 'threshold': probability_threshold,
                 'trained_at': datetime.now(timezone.utc).isoformat(),
                 'data_end_time': int(raw.iloc[-1]['open_time'])},
                MODEL_DIR / f'{symbol}_xgboost.joblib')
    with connect() as con:
        cursor = con.execute('INSERT INTO model_runs(symbol,created_at,train_rows,validation_rows,test_rows,metrics_json) VALUES (?,?,?,?,?,?)',
                             (symbol, datetime.now(timezone.utc).isoformat(), len(train), len(val), len(test), json.dumps(metrics)))
        run_id = cursor.lastrowid
        con.executemany('''INSERT INTO trades(run_id,symbol,signal_time,entry_time,exit_time,entry_price,exit_price,probability,result,net_return)
                           VALUES (:run_id,:symbol,:signal_time,:entry_time,:exit_time,:entry_price,:exit_price,:probability,:result,:net_return)''',
                        [{'run_id': run_id, 'symbol': symbol, **t} for t in trades])
    metrics['run_id'] = run_id
    return metrics


def latest_runs(limit=20):
    with connect() as con:
        rows = con.execute('SELECT id, metrics_json FROM model_runs ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
    return [{'run_id': row[0], **json.loads(row[1])} for row in rows]
