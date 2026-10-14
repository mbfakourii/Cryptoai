# Cryptoai — Shared paper wallet / 50 markets

## Shared 20 USDT virtual wallet

The scheduled workflow now performs automatic **paper** spot buying and selling across 50 USDT markets using **one shared 20 USDT balance**. It never places exchange orders or connects a real wallet. All positions, fees, cash, equity history and trades persist in `cryptoai-portfolio/paper/portfolio.json`; the local dashboard displays that remote wallet with a chart, open positions and trade history. This public state contains simulated balances only.

Run **Actions → Train 50-market portfolio → Run workflow** first, then **Portfolio candle-close timer → Run workflow**. The timer runs entirely on GitHub-hosted runners and dispatches **Paper portfolio to Telegram** at each 15-minute UTC candle boundary. Each cycle updates the durable online learners before scoring new entries; GitHub queue delays can postpone execution. Weekly baseline maintenance builds one pooled XGBoost model across the 50 highest-volume eligible markets with complete history, excluding stablecoins. Each market contributes 3,000 closed 15-minute candles. Chronological split boundaries are shared across markets, and future labels are purged at boundaries. The label target is a next-open to horizon-close return above 0.4%; reported Brier/AUC measure classification, not portfolio profit. Published model and pinned dependencies are stored atomically in `cryptoai-models/portfolio-bundle.zip`. The earlier BTC/ETH training workflow remains available for separate research backtests.

Settings live in `config/paper-trading.json`: initial balance 20, report interval 10 minutes, entry score 0.65, exit score 0.40, maximum three positions, maximum 35% equity per position, 10% cash reserve, 1% equity stop-distance sizing and a persistent 20% drawdown entry halt. The original balance is applied **only when no stored wallet exists**; changing the config does not reset saved capital. Binance quantity steps and minimum order notional are respected. A 20 USDT wallet may only afford one or two positions, and many signals can be skipped because the minimum order is too large.

The configured entry score is the upper bound for new portfolio models. Training selects an entry threshold from validation-score quantiles above the configured exit score, requiring at least 20 non-overlapping simulated validation trades with a positive mean return after fees and slippage. Candidates use each market's next-open quotes and scheduled stop/target/time/model-score exits, including a fixed 2 bps spread plus fees and slippage. They do not assume intrabar fills at a stop or target. This per-market diagnostic does not simulate the shared wallet's capital constraints or workflow delays. The test slice is only evaluated after selection and never chooses the threshold. If no candidate qualifies, the configured cutoff is retained for diagnostics and `threshold_selection.status` reports `no_validated_threshold`; new buys are disabled. Deployment also requires at least 20 held-out test trades with positive mean net return. The test is an acceptance gate, not a source for threshold tuning. Online scores cannot override this entry gate; existing positions can still exit. Models trained before this policy retain their fixed threshold until retrained. The runner uses the saved threshold consistently for prediction and order eligibility, and persists `entry_threshold` and `signal_scores` in the public paper wallet. Validation results do not guarantee future returns.

Simulated fills use current ask for buys and bid for sells, plus 0.02% slippage and 0.1% fee each side. Equity estimates liquidation after these costs. Stops, targets, lower model scores and a one-hour holding limit for new positions trigger exits at the current quote when the workflow runs. There is no continuous monitoring or assumed fill at a crossed stop: a delayed runner can exit at a worse price. Data errors suspend new buys while available positions can still exit. A model will wait for a candle after its training cutoff before making new entries.

Only a cycle that executes a paper BUY or SELL sends one aggregate Telegram message containing equity, cash, profit/loss, fees, positions and this cycle’s trades. Idle, HOLD and WAITING cycles still persist wallet state and refresh the dashboard without sending Telegram messages. Wallet writes use the GitHub Contents API’s SHA comparison; conflicting workers stop before sending. Reports are reserved durably **before** calling Telegram, so a crash or unknown send outcome is not automatically retried. Later cycles with new trades can send another notification. Trade decisions are processed once per closed candle, and rerunning the same ten-minute slot does not execute more trades. Keep `cryptoai-portfolio` intact; deleting it resets the simulated capital and duplicate protection.

The existing repository Secrets **TELEGRAM_BOT_TOKEN** and **TELEGRAM_CHAT_ID** are used unchanged. Edit them at **Settings → Secrets and variables → Actions**. `GITHUB_TOKEN` is automatic; allow updates to `cryptoai-models` and `cryptoai-portfolio` and `actions: write` for the timer. The timer dispatches learning and trading checks at fifteen-minute UTC boundaries without relying on GitHub's cron event delivery. Telegram notifications are triggered by executed paper trades, not by elapsed time. The legacy `report_interval_minutes` setting does not delay trade notifications.

All jobs use standard public-repository Linux runners, no paid data API, Actions cache or artifacts. Models and bounded wallet history live in repository branches. This aims to avoid metered Actions usage on GitHub Free. The project does not guarantee that the virtual wallet will gain value.

An executable **research-only** backend. It downloads closed public Binance candles, stores them in SQLite, builds past-only technical features, trains XGBoost with purged chronological train/validation/test splits, runs a long-only spot backtest, and serves saved results via FastAPI.

## Candle-by-candle learning and external context

The active prediction horizon is **one hour (4 closed 15-minute candles)**. After downloading the published model, the paper workflow checks its horizon and automatically retrains and publishes a matching model if its horizon or execution-validation version differs. Failed training/publication stops the job rather than trading with a mismatched model. This upgrade is one-time; subsequent cycles reuse the matching model and update the online learners.

The paper runner combines the saved XGBoost baseline with an incremental logistic classifier (one averaged, norm-bounded logistic-gradient update per matured candle). Every new closed 15-minute candle contributes a frozen observation for each scored market. Newly matured observations update the classifier on the next cycle; learned coefficients and pending observations persist atomically with the wallet under `learning`, including across GitHub runner restarts. The weekly job only refreshes the historical baseline: incremental learning does not wait for that job, and a full historical training run is not repeated every 15 minutes.

External inputs are CoinDesk and Cointelegraph RSS headlines from the preceding 24 hours, plus Yahoo Finance 15-minute charts for **GC=F (gold futures, not spot gold)** and **DX-Y.NYB (US Dollar Index, not USD/IRR)**. Price returns over one and four available bars, quote age, availability flags, news count and 64 hashed headline unigram/bigram features accompany the nine technical features and baseline score. This is statistical text-feature learning, not an LLM reading articles or a hand-written positive/negative headline rule. Yahoo's chart endpoint is an unofficial, unguaranteed integration and quotes may be delayed. Source errors and timestamps are visible in `market_context` and the dashboard.

News is frozen when observed, never backfilled into an earlier trading decision. Future-dated articles, unfinished macro candles and macro quotes older than four days are rejected. Weekend/delayed prices carry their actual timestamps and an age feature. Missing either macro series or both news feeds blocks new entries while existing stop, target, time and model exits remain available. No external API keys are required.

The market learner's label is a return greater than 0.4% from the ask observed when taking the snapshot to the close of the candle 4 bars after the signal candle. It cannot learn that outcome until that hour has elapsed. After this initial maturity delay, each new candle normally makes another batch of up to 50 observations available. It does not invent historic news or claim that repeatedly fitting the same candle creates new evidence. Missed source data is skipped or retained until the exact outcome candle is available; duplicate candles are not trained twice.

Before each update, frozen predictions are evaluated against their later outcomes. A 50/50 baseline/adaptive blend influences scores only after at least 200 matured observations spanning 96 evaluated signal candles and a lower exponentially weighted Brier score than the baseline on those observations. Each candle contributes one mean loss and one averaged gradient update, independent of market ordering or repeated identical rows. The step size is bounded by the batch feature norm. The evaluation decay is 0.99 per candle (roughly 69 candles half-life), rather than per symbol; 50 correlated markets are not treated as 50 independent time periods. Late-arriving samples from an already evaluated candle can train but do not count as another evaluation period. Until then it learns in shadow mode. This gate measures prediction error, not guaranteed net profitability or statistical significance. Learned headline coefficients start at zero, so external features need fresh labeled observations before they can influence predictions.

A separate online entry filter learns from each closed paper trade's net realized profit/loss, including execution costs, using features saved at entry. It can veto an entry only after at least 30 recorded trade outcomes and better Brier performance than a constant 0.5 prediction. It never learns an open trade as a completed win/loss, never rewrites risk limits and never forces an exit. Trades opened before this feature was installed are excluded because their point-in-time context was not recorded. This is supervised outcome feedback, not reinforcement learning.

Evaluation version 2 preserves existing learned weights, sample counts, pending observations and trade-entry snapshots. It resets only the old evaluation statistics, recording them in `evaluation_migration`; old frozen predictions can still train but cannot qualify the new evaluator. Fresh forward predictions must pass the gate again. This fixes update instability and order-sensitive evaluation, but does not establish profitability or guarantee that the adaptive model beats the baseline.

`learning_summary` exposes evaluation version and candle counts alongside sample counts, last update, active/shadow states and dropped observations. Pending history is bounded and trimmed if required to keep the wallet within its storage budget; this is reported, not silently reset. Invalid stored learner schemas fail explicitly. The 12-to-4-bar upgrade resets only the online classifiers and their pending labels, records migration counts, and preserves candle duplicate protection. Wallet capital, trade history and existing positions are retained; existing positions keep their original expiry. Other horizon changes require an explicit learner migration. The timer accepts old ten-minute handoffs during migration without treating their slot numbers as fifteen-minute slots.

## Reproducible strategy comparison

Run `python -m app.strategy_comparison` to compare the XGBoost baseline with a fixed trend/breakout policy. It fetches complete 15-minute history for 50 current liquid markets, uses the same chronological splits for both policies, and prints JSON diagnostics. It never writes the shared wallet, publishes a model, sends Telegram messages or places orders. The **Compare paper trading strategies** Actions workflow runs the same command with `--publish` to save only its diagnostics to `cryptoai-models/research/strategy-comparison.json`, including the code revision and data timestamp.

The fixed policy buys when the close exceeds the previous 20-bar high, EMA20 is above EMA50, and volume is at least its 20-bar average. A close below EMA20 triggers a model-style exit; both strategies also use the existing ATR stop/target and one-hour expiry. Both replay through the actual `paper_cycle` position/risk engine with identical 20 USDT starting capital, quantity rules, fees, slippage and assumed 2 bps spreads. Execution uses the next bar's open as the observed quote. Actual spreads and workflow delays can differ.

Entry thresholds for the model are selected on validation only. The comparison selects a strategy on validation net return with drawdown as the tie-breaker, requiring at least 20 closed trades, positive return and no risk halt. Test results are then reported without tuning either method against that period. Raw diagnostic results are also shown for rejected candidates; those are not an authorization to trade. The output explicitly identifies whether test results support the selected candidate. No candidate is automatically substituted into the live runner by this research command.

Historical news and incremental learning are excluded from this baseline comparison because no point-in-time news archive is available for that history. It is not a backtest of the full news-adaptive live system. The current-universe selection also has survivorship/selection limitations; repeatable positive results and forward paper evaluation are needed before interpreting this as a durable trading advantage.

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

For a private Telegram destination, open your own bot and press **Start**, then run this from the project directory:

```powershell
.\.venv\Scripts\python.exe -m app.telegram_setup
```

Enter the BotFather token at the hidden prompt. It stays in process memory and is not saved; the helper prints only private chat IDs found through `getUpdates`, not message contents or the token. If multiple users have contacted the bot, it lists multiple IDs: identify your own before configuring a destination. A bot with an active webhook cannot use `getUpdates`; do not remove an existing webhook blindly.

### GitHub setup

1. Upload the source to a **public** GitHub repository's default branch, including `.github/workflows/`. These workflows deliberately skip all jobs in private repositories, including manual runs, to avoid consuming private-repository minutes. A public repository exposes the source and research state; it does not expose Actions Secrets. `.gitignore` excludes the local database, models, virtual environment and real env files. There is no need to upload the local generated model binaries. If you prefer to keep the source private, use the local watcher instead.
2. In **Settings → Secrets and variables → Actions → New repository secret**, add:
   - `TELEGRAM_BOT_TOKEN`: the token obtained from Telegram's official `@BotFather` when creating your bot.
   - `TELEGRAM_CHAT_ID`: the destination chat/group/channel ID, including its minus sign if present. For a private chat, start your bot first; the Bot API `getUpdates` response contains `message.chat.id`. For a channel, add the bot as an administrator with posting rights; an eligible public channel username such as `@your_channel` is also supported by `sendMessage`.
3. Ensure repository/organization policy permits the trading workflow's `contents: write` permission and the timer's `actions: write` permission. `GITHUB_TOKEN` is provided automatically by GitHub; do not create a personal access token. Rulesets must permit creation and updates of `cryptoai-portfolio`, `cryptoai-signal-state` and `cryptoai-models` by GitHub Actions.
4. In **Actions → Train 50-market portfolio → Run workflow**, run training once on the default branch. The pooled portfolio model and its exact Python dependency versions are compressed into one atomic bundle in the `cryptoai-models` branch. Incomplete training does not replace the previous published model. Subsequent portfolio training runs are scheduled weekly, Monday at 01:43 UTC (05:13 Tehran). The separate **Train signal models** workflow trains only the legacy BTC/ETH research models.
5. In **Actions → Portfolio ten-minute timer → Run workflow**, start continuous checking on `main`, leaving `next_slot` empty and `cycles` at 30. The first cycle starts immediately, then checks are dispatched at minutes **00, 10, 20, 30, 40, 50** of every hour. Updates to the timer workflow or its Python source on `main` automatically bootstrap one cycle and hand off to the continuous timer. No local process, external scheduler, API subscription or extra secret is needed.
6. **Paper portfolio to Telegram** remains available for a single manual check. The runner downloads the last complete pooled portfolio model, installs its training dependency versions, and checks 50 markets against one shared virtual wallet. Telegram is sent only for executed paper buys or sells; idle cycles stay silent. The Actions run summary shows the cycle result. Missing models and missing secrets fail the run visibly; per-market errors appear in the cycle result and suspend new buys.

The timer holds one standard public Linux runner while waiting between dispatches. After at most 30 cycles (about five hours), it dispatches its successor with the next slot before exiting, below GitHub's six-hour job limit. A shared timer concurrency group allows only one active timer; the successor receives a fresh automatic token and reads the latest `main`. Trading jobs have their own concurrency group and remain capped at eight minutes. Dispatch duration does not shift the ten-minute boundaries, and delayed timer startups skip missed slots instead of replaying them. Temporary dispatch failures are logged and the next slot still runs; authorization failures stop the timer visibly.

To stop continuous checking, disable **Portfolio candle-close timer** and cancel its running or queued runs. To recover after a timer failure or GitHub outage, enable it and use **Run workflow** again. Cancelling an active timer stops its continuation. Runner availability and API outages can still delay or interrupt checks; the timer bypasses cron delivery but cannot guarantee uninterrupted GitHub service. A new signal is possible only after a new 15-minute candle closes, even though checks run every 10 minutes. Training is separate; models are not retrained every 10 minutes. Hosted runners must be able to reach Binance and Telegram; regional/API failures are reported rather than substituted with fabricated data.

### Duplicate suppression and failure handling

Immediately after training, the newest closed candle may still belong to the training dataset. The runner reports `WAITING` successfully without scoring or sending until a later candle closes; invalid model metadata and stale or missing market data remain errors.

The delivery key is `(symbol, candle open timestamp)`; changing the model does not bypass it. Before contacting Telegram, the runner durably reserves that key. Local workers use an atomic SQLite unique constraint. GitHub runs use a create-if-absent JSON file in the **`cryptoai-signal-state`** branch via the GitHub Contents API, plus a shared concurrency group. This state survives fresh runners, reruns and retraining. Do not delete/reset the state branch or database: doing so removes duplicate protection. A local sender and GitHub sender have separate state; use only one sender for the same destination.

This is **at-most-once automatic sending**, not guaranteed exactly-once delivery. If the process crashes after reservation, Telegram times out, or Telegram rejects a message, that candle is not automatically resent. This favors avoiding duplicates over guaranteed delivery. Inspect `signal-state/<symbol>/<timestamp>.json` in the state branch: `reserved` or `unknown_or_failed` requires manual investigation; `sent` means Telegram acknowledged the message. Confirm whether the message arrived before making any manual resend. If the state service cannot confirm a reservation, Telegram is not called.

The workflows use only standard `ubuntu-latest` runners in public repositories. They upload **no Actions artifacts**, enable **no dependency cache**, use no paid API, and publish no paid hosting. Model bundles are capped at 500 KB per publication and stored in repository Git history. Public-repository standard runner minutes are free; private jobs are skipped. This setup adds no metered Actions storage, though other repositories or services on your account can still incur charges. For an account-wide safeguard, set a GitHub Actions budget with **Stop usage when budget limit is reached**; this code does not change account billing settings.

The scheduled runner does not host FastAPI. The dashboard’s paper-wallet section reads public GitHub wallet state through /portfolio; the BTC/ETH research sections continue to display local SQLite evaluations. Inspect GitHub workflow output and delivery JSON in the state branch for remote results. Model and signal state contain no bot token. Preview mode does not send any messages.

References: [GitHub workflow dispatch](https://docs.github.com/en/rest/actions/workflows#create-a-workflow-dispatch-event), [Triggering workflows with GITHUB_TOKEN](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow), [GitHub Actions limits](https://docs.github.com/en/actions/reference/limits), [GitHub Contents API](https://docs.github.com/en/rest/repos/contents#create-or-update-file-contents), [Telegram sendMessage](https://core.telegram.org/bots/api#sendmessage).

Market data requests use Binance's official public-data-only endpoint, `https://data-api.binance.vision`, for both historical candles and live server time/candles. This avoids relying on the trading API host for public market data on hosted runners.

## Cautions / known limitations

- Public Binance API availability depends on region and network. HTTP 429/451 and network errors are possible; respect exchange rate limits and terms.
- `fetch` backfills candles backwards in batches; missing intervals trigger a training error rather than silently hiding gaps.
- Re-running training on the same dataset does not provide new out-of-sample evidence.
- No probability calibration, walk-forward refits, market-depth liquidity modeling, or regime shift checks yet.
- Backtest entries at the exact next open are optimistic under some execution conditions.
- Never treat a backtest as a prediction of future profits.
