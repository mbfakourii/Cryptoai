import argparse
import json
from app.core import connect, fetch_candles, train_and_backtest


def main():
    parser = argparse.ArgumentParser(description='Crypto AI Signal research CLI')
    parser.add_argument('command', choices=['init-db', 'fetch', 'train'])
    parser.add_argument('--symbol', choices=['BTCUSDT', 'ETHUSDT'], default='BTCUSDT')
    parser.add_argument('--limit', type=int, default=5000)
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--horizon', type=int, default=12)
    args = parser.parse_args()
    if args.command == 'init-db':
        with connect():
            pass
        print('Database initialized')
    elif args.command == 'fetch':
        print(f'Fetched {fetch_candles(args.symbol, args.limit)} closed candles')
    else:
        if not 0.5 <= args.threshold <= 1:
            parser.error('--threshold must be between 0.5 and 1')
        if not 1 <= args.horizon <= 96:
            parser.error('--horizon must be between 1 and 96')
        print(json.dumps(train_and_backtest(args.symbol, args.horizon, args.threshold), indent=2))

if __name__ == '__main__':
    main()
