# Crypto AI Signal v1 — Spot / 15m

An executable **research-only** backend. It downloads closed public Binance candles, stores them in SQLite, builds past-only technical features, trains XGBoost with purged chronological train/validation/test splits, runs a long-only spot backtest, and serves saved results via FastAPI.

## Setup (Python 3.11 or 3.12 recommended)

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.cli init-db
python -m app.cli fetch --symbol BTCUSDT --limit 10000
python -m app.cli fetch --symbol ETHUSDT --limit 10000
python -m app.cli train --symbol BTCUSDT --horizon 12 --threshold 0.65
python -m app.cli train --symbol ETHUSDT --horizon 12 --threshold 0.65
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the Persian dashboard. It displays saved BTC/ETH training runs, compounded equity, drawdown, per-trade returns, and filterable trades with CSV export. Dates use the Tehran timezone. Saved results refresh every 60 seconds; this does not fetch market candles, train a model, or generate live signals.

The API remains available at `/docs`, `/health`, `/runs`, and `/trades/1`.

Run tests: `pytest -q`.

## Database

- `candles`: (symbol, open_time) unique; OHLCV in UTC milliseconds
- `model_runs`: one row per training/backtest run and metrics JSON
- `trades`: historical simulated entries/exits and net returns

The SQLite file is created at `data/crypto.db`, models under `data/models/`.

## Model / backtest assumptions

- Feature values at a signal candle's close use current/past closed candles only.
- Binary label: return from **next candle open** to close after `horizon` candles exceeds 0.2%. This is a directional research target, **not** a guaranteed profitable trade.
- Chronological 60/20/20 split with `horizon` rows purged at the train and validation boundaries. Validation is diagnostic; thresholds are fixed in advance, not optimized on test.
- Long-only; signal at candle close, buy next candle open; no overlapping positions per symbol; ATR-based SL/TP; stop is assumed first when stop and target both touch in the same candle.
- Fees: 10 basis points each side; slippage: 2 basis points each side. Real fills may be worse; gaps through stops are not fully modeled.
- Backtest trade simulation runs **within the test slice**; the first test signal has its entry in the next test candle.
- Metrics are test-set-only; reported win rate is **net profitable trades**, not TP-hit rate.
- The model is not calibrated; 0.65 is a *raw model score*, not a verified 65% chance.
- There is no real order execution or production auth. Live evaluations and optional Telegram research alerts are separate from the historical backtest.

## Live signals and Telegram

The dashboard's live section reads `/signals`. A separate evaluator fetches the latest 1,000 **closed** Binance spot candles for BTCUSDT and ETHUSDT, verifies freshness and continuous 15-minute intervals, and uses the saved XGBoost models. It emits a research `BUY` when the raw model score reaches the model's saved threshold (default 0.65); otherwise it records `HOLD` without sending a message. Stop and target references use the latest close and ATR, not an actual entry fill. There is no position management or real order execution.

Models must include the new symbol, interval and training-cutoff metadata. Retrain older model files once with the existing `train` commands. The evaluator refuses to score a candle within the model's training-data window; the first evaluation after training may need to wait for another candle to close.

Preview once, without sending or consuming delivery reservations:

```bash
python -m app.signals
```

Run locally every 10 minutes, updating the local dashboard:

```bash
python -m app.signals --watch --interval-seconds 600
```

To send locally, set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the process environment, then run:

```bash
python -m app.signals --send --watch --interval-seconds 600
```

`.env.example` lists the names only. This application does **not** automatically load `.env` files. Never commit the token or paste it into chat, logs, or workflow YAML. Local delivery state is stored in `data/crypto.db`; keep that database across restarts.

### GitHub setup

1. Upload the source to a **public** GitHub repository's default branch, including `.github/workflows/`. These workflows deliberately skip all jobs in private repositories, including manual runs, to avoid consuming private-repository minutes. A public repository exposes the source and research state; it does not expose Actions Secrets. `.gitignore` excludes the local database, models, virtual environment and real env files. There is no need to upload the local generated model binaries. If you prefer to keep the source private, use the local watcher instead.
2. In **Settings → Secrets and variables → Actions → New repository secret**, add:
   - `TELEGRAM_BOT_TOKEN`: the token obtained from Telegram's official `@BotFather` when creating your bot.
   - `TELEGRAM_CHAT_ID`: the destination chat/group/channel ID, including its minus sign if present. For a private chat, start your bot first; the Bot API `getUpdates` response contains `message.chat.id`. For a channel, add the bot as an administrator with posting rights; an eligible public channel username such as `@your_channel` is also supported by `sendMessage`.
3. Ensure repository/organization policy permits the workflow's `contents: write` permission. `GITHUB_TOKEN` is provided automatically by GitHub; do not create a personal access token. Rulesets must permit creation and updates of `cryptoai-signal-state` and `cryptoai-models` by GitHub Actions.
4. In **Actions → Train signal models → Run workflow**, run training once on the default branch. Both models and their exact Python dependency versions are compressed into one atomic bundle in the `cryptoai-models` branch. Incomplete training does not replace the previous published pair. Subsequent training runs are scheduled weekly, Monday at 02:23 UTC (05:53 Tehran).
5. In **Actions → Live signals to Telegram → Run workflow**, test the runner. Its schedule runs at minutes **03, 13, 23, 33, 43, 53** of every hour. The runner downloads the last complete published model pair, installs the training dependency versions, and checks both symbols. Missing models, market failures and missing secrets stop sending and fail the run visibly.

GitHub schedules can be delayed and are not a real-time SLA. A new signal is possible only after a new 15-minute candle closes, even though checks run every 10 minutes. Training is separate; models are not retrained every 10 minutes. Hosted runners must be able to reach Binance and Telegram; regional/API failures are reported rather than substituted with fabricated data.

### Duplicate suppression and failure handling

The delivery key is `(symbol, candle open timestamp)`; changing the model does not bypass it. Before contacting Telegram, the runner durably reserves that key. Local workers use an atomic SQLite unique constraint. GitHub runs use a create-if-absent JSON file in the **`cryptoai-signal-state`** branch via the GitHub Contents API, plus a shared concurrency group. This state survives fresh runners, reruns and retraining. Do not delete/reset the state branch or database: doing so removes duplicate protection. A local sender and GitHub sender have separate state; use only one sender for the same destination.

This is **at-most-once automatic sending**, not guaranteed exactly-once delivery. If the process crashes after reservation, Telegram times out, or Telegram rejects a message, that candle is not automatically resent. This favors avoiding duplicates over guaranteed delivery. Inspect `signal-state/<symbol>/<timestamp>.json` in the state branch: `reserved` or `unknown_or_failed` requires manual investigation; `sent` means Telegram acknowledged the message. Confirm whether the message arrived before making any manual resend. If the state service cannot confirm a reservation, Telegram is not called.

The workflows use only standard `ubuntu-latest` runners in public repositories. They upload **no Actions artifacts**, enable **no dependency cache**, use no paid API, and publish no paid hosting. Model bundles are capped at 500 KB per publication and stored in repository Git history. Public-repository standard runner minutes are free; private jobs are skipped. This setup adds no metered Actions storage, though other repositories or services on your account can still incur charges. For an account-wide safeguard, set a GitHub Actions budget with **Stop usage when budget limit is reached**; this code does not change account billing settings.

The scheduled runner does not synchronize its SQLite database to your local dashboard or host FastAPI; the local dashboard displays local evaluations. Inspect GitHub workflow output and delivery JSON in the state branch for remote results. Model and signal state contain no bot token. Preview mode does not send any messages.

References: [GitHub schedules](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onschedule), [GitHub Contents API](https://docs.github.com/en/rest/repos/contents#create-or-update-file-contents), [Telegram sendMessage](https://core.telegram.org/bots/api#sendmessage).

## Cautions / known limitations

- Public Binance API availability depends on region and network. HTTP 429/451 and network errors are possible; respect exchange rate limits and terms.
- `fetch` backfills candles backwards in batches; missing intervals trigger a training error rather than silently hiding gaps.
- Re-running training on the same dataset does not provide new out-of-sample evidence.
- No probability calibration, walk-forward refits, portfolio-level cash allocation, liquidity modeling, or regime shift checks yet.
- Backtest entries at the exact next open are optimistic under some execution conditions.
- Never treat a backtest as a prediction of future profits.
