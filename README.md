# Cryptoai — Shared paper wallet / 50 markets

## Shared 20 USDT virtual wallet

The scheduled workflow now performs automatic **paper** spot buying and selling across 50 USDT markets using **one shared 20 USDT balance**. It never places exchange orders or connects a real wallet. All positions, fees, cash, equity history and trades persist in `cryptoai-portfolio/paper/portfolio.json`; the local dashboard displays that remote wallet with a chart, open positions and trade history. This public state contains simulated balances only.

Run **Actions → Train 50-market portfolio → Run workflow** first, then **Portfolio candle-close timer → Run workflow**. The timer runs entirely on GitHub-hosted runners and dispatches **Paper portfolio to Telegram** at each 15-minute UTC candle boundary. Each cycle updates the durable online learners before scoring new entries; GitHub queue delays can postpone execution. Weekly baseline maintenance builds one pooled XGBoost model across the 50 highest-volume eligible markets with complete history, excluding stablecoins. Each market contributes 3,000 closed 15-minute candles. Chronological split boundaries are shared across markets, and future labels are purged at boundaries. The label target is a next-open to horizon-close return above 0.4%; reported Brier/AUC measure classification, not portfolio profit. Published model and pinned dependencies are stored atomically in `cryptoai-models/portfolio-bundle.zip`. The earlier BTC/ETH training workflow remains available for separate research backtests.

Settings live in `config/paper-trading.json`: initial balance 20, report interval 10 minutes, entry score 0.65, exit score 0.40, maximum three positions, maximum 35% equity per position, 10% cash reserve, 1% equity stop-distance sizing and a persistent 20% drawdown entry halt. The original balance is applied **only when no stored wallet exists**; changing the config does not reset saved capital. Binance quantity steps and minimum order notional are respected. A 20 USDT wallet may only afford one or two positions, and many signals can be skipped because the minimum order is too large.

The configured `paper_entry_policy` is `after_training`: a complete model with
matching schema, horizon and scheduled-execution metadata can authorize paper
entries even when profitability validation fails. This applies only to the
virtual wallet. Prediction thresholds, fresh-candle checks, market/context
availability, feedback vetoes, position limits and the persistent drawdown halt
still apply. Existing successfully trained models qualify without retraining.
The dashboard and trade reports label this permission as experimental;
`entry_policy_approved` continues to report profitability validation honestly,
while `paper_entries_allowed` and `paper_entry_status` report paper permission.
Set `paper_entry_policy` to `validated` to restore the profitability requirement;
omitting the setting also retains the strict behavior.

Under the strict `validated` policy, the configured entry score is the upper bound for new portfolio models. Training selects an entry threshold from validation-score quantiles above the configured exit score. Every candidate uses one shared wallet and the actual `paper_cycle` engine: ranked simultaneous signals compete for the same cash, quantity/minimum-order rules, reserves, position limits and risk budget. Selection requires at least 20 closed trades, positive account return after costs, no risk halt and no unfinished positions. Candidates are ranked by account return, then drawdown and threshold. Synchronized next-open quotes include a fixed 2 bps spread plus fees and slippage; exits are observed stop/target/time/model-score exits, without assumed intrabar fills. Entries stop early enough to drain positions inside each split. Workflow delays and historical changes in market filters remain unmodeled.

The test slice is evaluated only after selection and never chooses the threshold. Strict baseline approval also requires at least 20 held-out test trades with positive shared-wallet return, no halt and no unfinished positions. If no validation candidate qualifies, the configured cutoff is retained for diagnostics and `threshold_selection.status` reports `no_validated_threshold`. `shared_wallet_v1` identifies this evaluation separately from execution semantics: old independent-market results cannot authorize strict entries, but previously trained models retain explicit `after_training` paper permission and their saved threshold. Installing this evaluator alone does not retrain the model or reset online evidence; the next baseline training uses it. An independently qualified online policy can authorize adaptive entries without weekly retraining. Validation results do not guarantee future returns.

Published models retain the chronological training/validation/test separation rather than refitting on held-out data and inheriting its approval. `metrics.data_windows` records the feature and label-availability endpoints of every split; `training_feature_end_time` and `training_label_available_through_time` identify the actual fitted-data cutoff. `data_end_time` is the end of all collected history, including evaluation data, and is not the training cutoff. The dashboard distinguishes the training-data endpoint from the model build time and reports shared-wallet test return when available.

Simulated fills use current ask for buys and bid for sells, plus 0.02% slippage and 0.1% fee each side. Equity estimates liquidation after these costs. Stops, targets, lower model scores and a one-hour holding limit for new positions trigger exits at the current quote when the workflow runs. There is no continuous monitoring or assumed fill at a crossed stop: a delayed runner can exit at a worse price. Data errors suspend new buys while available positions can still exit. A model waits for a candle after the end of its collected training/evaluation history before making new entries.

Only a cycle that executes a paper BUY or SELL sends one aggregate Telegram message containing equity, cash, profit/loss, fees, positions and this cycle’s trades. Idle, HOLD and WAITING cycles still persist wallet state and refresh the dashboard without sending Telegram messages. Wallet writes use the GitHub Contents API’s SHA comparison; conflicting workers stop before sending. Reports are reserved durably **before** calling Telegram, so a crash or unknown send outcome is not automatically retried. Later cycles with new trades can send another notification. Trade decisions are processed once per closed candle, and rerunning the same ten-minute slot does not execute more trades. Keep `cryptoai-portfolio` intact; deleting it resets the simulated capital and duplicate protection.

The existing repository Secrets **TELEGRAM_BOT_TOKEN** and **TELEGRAM_CHAT_ID** are used unchanged. Edit them at **Settings → Secrets and variables → Actions**. `GITHUB_TOKEN` is automatic; allow updates to `cryptoai-models` and `cryptoai-portfolio` and `actions: write` for the timer. The timer dispatches learning and trading checks at fifteen-minute UTC boundaries without relying on GitHub's cron event delivery. Telegram notifications are triggered by executed paper trades, not by elapsed time. The legacy `report_interval_minutes` setting does not delay trade notifications.

All jobs use standard public-repository Linux runners, no paid data API, Actions cache or artifacts. Models and bounded wallet history live in repository branches. This aims to avoid metered Actions usage on GitHub Free. The project does not guarantee that the virtual wallet will gain value.

An executable **research-only** backend. It downloads closed public Binance candles, stores them in SQLite, builds past-only technical features, trains XGBoost with purged chronological train/validation/test splits, runs a long-only spot backtest, and serves saved results via FastAPI.

## Candle-by-candle learning and external context

The active prediction horizon is **one hour (4 closed 15-minute candles)**. After downloading the published model, the paper workflow checks its horizon and automatically retrains and publishes a matching model if its horizon or execution-policy version differs. Failed training/publication stops the job rather than trading with a mismatched model. This upgrade is one-time; subsequent cycles reuse the matching model and update the online learners. A profitability-evaluation version change alone does not trigger this retraining path.

The paper runner combines the saved XGBoost baseline with an incremental logistic classifier (one averaged, norm-bounded gradient update per matured candle). Every new closed 15-minute candle contributes a frozen observation for each scored market. Newly matured observations update the classifier on the next cycle; learned coefficients and pending observations persist atomically with the wallet under `learning`, including across GitHub runner restarts. The weekly job only refreshes the historical baseline: incremental learning does not wait for that job, and a full historical training run is not repeated every 15 minutes.

External inputs are CoinDesk and Cointelegraph RSS headlines from the preceding 24 hours, plus Yahoo Finance 15-minute charts for **GC=F (gold futures, not spot gold)** and **DX-Y.NYB (US Dollar Index, not USD/IRR)**. Price returns over one and four available bars, quote age, availability flags, news count and 64 hashed headline unigram/bigram features accompany the nine technical features and baseline score. This is statistical text-feature learning, not an LLM reading articles or a hand-written positive/negative headline rule. Yahoo's chart endpoint is an unofficial, unguaranteed integration and quotes may be delayed. Source errors and timestamps are visible in `market_context` and the dashboard.

News is frozen when observed, never backfilled into an earlier trading decision. Future-dated articles, unfinished macro candles and macro quotes older than four days are rejected. Weekend/delayed prices carry their actual timestamps and an age feature. Missing either macro series or both news feeds blocks new entries while existing stop, target, time and model exits remain available. No external API keys are required.

The market learner's label is a return greater than 0.4% from the ask observed when taking the snapshot to the close of the candle 4 bars after the signal candle. It cannot learn that outcome until that hour has elapsed. After this initial maturity delay, each new candle normally makes another batch of up to 50 observations available. It does not invent historic news or claim that repeatedly fitting the same candle creates new evidence. Missed source data is skipped or retained until the exact outcome candle is available; duplicate candles are not trained twice.

Before each update, frozen predictions are evaluated against their later outcomes. A 50/50 baseline/adaptive blend influences scores only after at least 200 matured observations spanning 96 evaluated signal candles and a lower exponentially weighted Brier score than the baseline on those observations. Each candle contributes one mean loss and one averaged gradient update, independent of market ordering or repeated identical rows. The step size is bounded by the batch feature norm. The evaluation uses a normalized decayed sum of losses and weights, with decay 0.99 per candle (roughly 69 candles half-life), so the first candle does not receive an artificial initial weight, rather than per symbol; 50 correlated markets are not treated as 50 independent time periods. Late-arriving samples from an already evaluated candle can train but do not count as another evaluation period. Until then it learns in shadow mode. This gate measures prediction error, not guaranteed net profitability or statistical significance. Learned headline coefficients start at zero, so external features need fresh labeled observations before they can influence predictions.

The adaptive policy searches entry thresholds independently of the baseline. Separate durable shadow wallets under `learning.threshold_search` evaluate a predefined grid of 0.45, 0.50, 0.55, 0.60 and the baseline cutoff, excluding values outside the configured exit/entry bounds. This grid is a research hypothesis, not an assumed profitable range. Every wallet uses the same `paper_cycle` engine, observed bid/ask, fees, slippage, sizing, minimum order sizes, exits and one-hour expiry. They do not affect the main balance, notifications or actual-trade feedback. Selection requires 20 recent closed trades with positive summed net PnL, positive cumulative realized and liquidation PnL, and no halt. A qualifying threshold is frozen, and a fresh confirmation wallet must pass those same conditions on 20 subsequent closed trades. Selection trades never qualify confirmation. Only confirmation plus the 96-candle prediction-quality gate can authorize main-wallet adaptive entries. The main execution engine then uses the confirmed threshold, and all normal risk and context guards remain in force. Confirmation is reevaluated every cycle; failure revokes adaptive entry authorization. An independently validated baseline remains available. No weekly retrain is needed.

The search identity includes the baseline revision, evaluation version and execution config. The main-wallet `paper_entry_policy` switch is excluded, so changing it preserves shadow evidence. Identity changes reset shadow evidence without resetting the main wallet or learned coefficients. Each shadow wallet has its own persistent drawdown halt. Recent profit evidence expires after seven days. `online_entry_policy` reports the selection/confirmation phase, per-threshold open and closed trades, profits and the selected threshold; `baseline_entry_policy_approved` reports the separate baseline gate. Candidate counts are policy trials, not distinct main-wallet trades. No threshold is forced to trade or automatically relaxed after losses.

A confirmation that hits its risk halt or fails the economic criteria after 20 recent closes is archived. Legacy confirmation with at least 20 lifetime closes and nonpositive cumulative economics can also expire after seven days. Selection restarts if every candidate is halted; partial candidate failures retain their evidence. Recovery enters `retry_wait` for at least 15 minutes and requires healthy data and a strictly newer candle. New selection and confirmation wallets use only subsequent signals: failed trades cannot qualify either stage. Existing evidence migrates in place, and temporary feed/context errors, worse Brier scores or merely aging profitable evidence do not restart the search. The main wallet's balance, positions and persistent halt, learned coefficients and independent baseline permission are preserved. Separate archives retain the last five failed selections and confirmations with bounded trade evidence; retry counters and reasons remain visible in `online_entry_policy` and the dashboard. Repeated trials do not establish statistical certainty of profit.

A separate online entry filter learns from each closed paper trade's net realized profit/loss, including execution costs, using features saved at entry. It can veto an entry only after at least 30 recorded trade outcomes and better Brier performance than a constant 0.5 prediction. It never learns an open trade as a completed win/loss, never rewrites risk limits and never forces an exit. Trades opened before this feature was installed are excluded because their point-in-time context was not recorded. This is supervised outcome feedback, not reinforcement learning.

Evaluation version 3 trains the market learner against squared error of the actual 50/50 deployed blend, rather than log loss of the unblended learner. Regularization anchors market coefficients toward the baseline prior. Trade feedback continues to optimize its own logistic prediction. Brier diagnostics use a normalized exponentially weighted average rather than seeding an entire unit of weight with the first observed loss. Existing weights, lifetime sample counts, pending samples and trade-entry snapshots are retained. Prior metrics and evaluated candle counts are archived in `evaluation_migration`. Fresh frozen predictions must qualify the revised 96-candle evaluator; old predictions can still train but cannot qualify it. The prior evaluation is not retroactively relabeled as evidence for the new training rule. These changes correct objective and evaluation mismatches but do not guarantee better future forecasts or profit.

`learning_summary` exposes evaluation version and candle counts alongside sample counts, last update, active/shadow states and dropped observations. Pending history is bounded and trimmed if required to keep the wallet within its storage budget; this is reported, not silently reset. Invalid stored learner schemas fail explicitly. The 12-to-4-bar upgrade resets only the online classifiers and their pending labels, records migration counts, and preserves candle duplicate protection. Wallet capital, trade history and existing positions are retained; existing positions keep their original expiry. Other horizon changes require an explicit learner migration. The timer accepts old ten-minute handoffs during migration without treating their slot numbers as fifteen-minute slots.

## Reproducible strategy comparison

`python -m app.btc_walk_forward --start 2024-10-01 --end 2026-10-01 --output research/btc-walk-forward-2024-2026.json`
runs the longer BTC comparison.
Dates are UTC and the end is exclusive. Downloads are validated and cached in
`data/research/`; retrying preserves completed 30-day chunks. The report includes
the input CSV checksum, runtime versions, configuration and market filters.

Each fold trains on 180 preceding days, selects settings on the next 30 days,
and tests them on the following 60 days (a final partial fold needs 30 days).
Training labels are purged at the validation boundary; trades drain before
validation/test boundaries. The BTC-only baseline classifier, its causal online
blend, a net-return regressor and trend/breakout share the actual paper engine,
20 USDT wallet and execution costs. All families may select a 1/2/4-hour holding
cap on validation while forecast labels remain one hour. The online blend learns
only after labels mature; other models stay frozen within each fold.

Reports distinguish the best validation-selected diagnostic from the gated
policy, which stays in cash when validation has fewer than 20 closed trades,
nonpositive net return or a risk halt. Test results never choose parameters.
Every fold restarts its wallet, so compounded returns normalize independent
folds rather than claim one continuously executable account history. Higher-cost
tests reuse the same frozen policy. These technical-only BTC models are proxies,
not a reconstruction of the deployed 50-market model with historical news.
The command does not update online state or activate purchases.

The checked-in [2024–2026 result](research/btc-walk-forward-2024-2026.json)
uses 70,080 candles and nine test folds from 2025-04-29 through 2026-09-30.
The net-return model had the smallest diagnostic loss: -1.80% normalized
compounded return across 51 closed trades, versus -47.59% for the adaptive
classifier, -49.29% for the baseline and -52.70% for trend/breakout. These are
independently restarted test wallets, including rejected validation candidates;
they are not returns of the online account or approved trading policies.

The net-return model passed the 20-trade profitable validation requirement in
zero folds, so its gated policy stayed in cash. Under higher execution costs its
diagnostic return fell to -4.07%. No tested family established overall
profitability. Removing all candles after the first test fold reproduced every
first-fold validation candidate, chosen policy and test result exactly.

For a fast single-market diagnostic using the local closed-candle archive, run
`python -m app.single_market_replay --symbol BTCUSDT` (or `ETHUSDT`). This reads
SQLite and saved market metadata without changing the database, model bundle or
remote wallet. It fits a baseline on the chronological training slice, warms the
online learner on validation and updates it only after each forecast horizon
matures. Entry quantiles and the lower exit quantile come from validation scores
only, so a BTC score distribution below 0.45 can be investigated without copying
the baseline's score cutoffs. The fixed 0.65/0.40 policy remains a comparison.
Only a candidate with at least 20 validation trades and positive net return can
qualify; its later test result cannot select or tune the policy. If none qualifies,
the best validation candidate with enough trades is reported as a rejected
diagnostic, never as approval. This technical-only replay excludes historical
news and macro inputs and does not substitute for fresh forward confirmation.
It has no publication or live-activation option.

`python -m app.btc_strategy_research` investigates the source of BTC losses with
a separate net-return regressor and nine entry/exit hypotheses with one-, two-
and four-hour holding caps (27 combinations). The forecast target stays one hour.
Its target reproduces observed scheduled ATR stop/target and time exits after
fees, slippage and spread. Candidate selection uses the validation slice only;
the later archive slice and newly fetched candles after the archive are reported
separately. Cost-free and higher-cost replays expose sensitivity to execution
costs. Two profitable trades cannot qualify a model: validation, test, cost stress
and fresh data must each meet the minimum evidence and profitability checks.
Forecast errors are compared with a constant training-mean prediction; the best
validation regressor policy is also reported separately even when it fails the
evidence gate. Reused archive and post-archive data are exploratory, not fresh
forward confirmation. This diagnostic neither publishes changes nor places orders.

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
