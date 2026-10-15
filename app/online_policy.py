import copy

from app.core import INTERVAL_MS
from app.online_learning import active
from app.portfolio import new_wallet, paper_cycle

MIN_CLOSED_TRADES = 20
EVIDENCE_MAX_AGE_MS = 7 * 86400000
MAX_FAILED_CONFIRMATIONS = 5
MAX_FAILED_SELECTIONS = 5


def confirmation_failure(summary, selected_at, now):
    if summary['halted']:
        return 'risk_halt'
    if summary['profitable_after_costs']:
        return None
    if summary['recent_closed_trades'] >= MIN_CLOSED_TRADES:
        return 'unprofitable_confirmation'
    if (now - selected_at >= EVIDENCE_MAX_AGE_MS and summary['closed_trades'] >= MIN_CLOSED_TRADES
            and (summary['realized_net_pnl'] <= 0 or summary['total_net_pnl'] <= 0)):
        return 'expired_unprofitable_confirmation'
    return None


def archive_failed_confirmation(search, reason, candle, now):
    confirmation = search['confirmation']
    wallet = confirmation['wallet']
    failure = {'phase': 'confirmation', 'reason': reason, 'failed_at': now, 'failed_candle': candle,
               'selected_at': search['selected_at'], 'selected_threshold': search['selected_threshold'],
               'summary': copy.deepcopy(confirmation['summary']),
               'recent_closed_trades': copy.deepcopy([
                   event for event in wallet['trade_history'] if event['side'] == 'SELL'
               ][-MIN_CLOSED_TRADES:]),
               'open_positions': copy.deepcopy(wallet['positions']),
               'selection': {key: copy.deepcopy(value['summary'])
                             for key, value in search['candidates'].items() if value is not None}}
    search['failed_confirmations'] = (search['failed_confirmations'] + [failure])[-MAX_FAILED_CONFIRMATIONS:]
    wait_for_fresh_selection(search, reason, candle, now)


def archive_exhausted_selection(search, candle, now):
    retained_trades_per_candidate = MIN_CLOSED_TRADES // len(search['candidates'])
    candidates = {}
    for key, candidate in search['candidates'].items():
        wallet = candidate['wallet']
        candidates[key] = {'summary': copy.deepcopy(candidate['summary']),
                           'recent_closed_trades': copy.deepcopy([
                               event for event in wallet['trade_history'] if event['side'] == 'SELL'
                           ][-retained_trades_per_candidate:]),
                           'open_positions': copy.deepcopy(wallet['positions'])}
    failure = {'phase': 'selection', 'reason': 'selection_risk_halt',
               'failed_at': now, 'failed_candle': candle, 'candidates': candidates}
    search['failed_selections'] = (search['failed_selections'] + [failure])[-MAX_FAILED_SELECTIONS:]
    search['selection_retry_count'] += 1
    wait_for_fresh_selection(search, 'selection_risk_halt', candle, now)


def wait_for_fresh_selection(search, reason, candle, now):
    search.update(phase='retry_wait', retry_count=search['retry_count'] + 1,
                  retry_reason=reason, retry_after=now + INTERVAL_MS,
                  evidence_after_candle=candle)


def signals_after(signals, candle):
    filtered = copy.deepcopy(signals)
    if candle is not None:
        for signal in filtered.values():
            if signal.get('candle_time', -1) <= candle and signal['action'] in ('BUY', 'HOLD'):
                signal['action'] = 'WAITING'
    return filtered


def evaluate_fixed_policy(learning, signals, quotes, markets, errors, context_ready,
                           now, config, baseline_revision, baseline_entry_allowed):
    config = {key: value for key, value in config.items() if key != 'paper_entry_policy'}
    identity = {'config': config, 'baseline_revision': baseline_revision,
                'evaluation_version': learning['evaluation_version']}
    stored = learning.get('entry_policy')
    if stored is None or stored['identity'] != identity:
        stored = {'identity': copy.deepcopy(identity), 'wallet': new_wallet(config)}
    shadow_signals = copy.deepcopy(signals)
    for signal in shadow_signals.values():
        if signal['action'] not in ('BUY', 'HOLD'):
            continue
        if 'shadow_model_score' not in signal:
            signal['action'] = 'HOLD'
            continue
        signal['model_score'] = signal['shadow_model_score']
        signal['action'] = ('BUY' if context_ready and signal.get('feedback_entry_allowed', True)
                            and signal['model_score'] >= config['entry_threshold'] else 'HOLD')
    wallet, _ = paper_cycle(stored['wallet'], shadow_signals, quotes, markets, errors, now, config)
    wallet['trade_history'] = wallet['trade_history'][-200:]
    wallet['equity_history'] = wallet['equity_history'][-2:]
    closed = [event for event in wallet['trade_history']
              if event['side'] == 'SELL' and now - event['time'] <= EVIDENCE_MAX_AGE_MS][-MIN_CLOSED_TRADES:]
    recent_pnl = sum(event['pnl'] for event in closed)
    better = active(learning['market'], 200)
    profitable = (len(closed) >= MIN_CLOSED_TRADES and recent_pnl > 0
                  and wallet['realized_pnl'] > 0 and wallet['total_pnl'] > 0)
    approved = (better and profitable and not wallet['halted']
                and not wallet['stale_positions'] and not errors and context_ready)
    summary = {'approved': approved, 'better_than_baseline': better,
               'profitable_after_costs': profitable, 'closed_trades': wallet['closed_count'],
               'recent_closed_trades': len(closed), 'minimum_closed_trades': MIN_CLOSED_TRADES,
               'recent_net_pnl': recent_pnl, 'realized_net_pnl': wallet['realized_pnl'],
               'total_net_pnl': wallet['total_pnl'], 'halted': wallet['halted'],
               'updated_at': now, 'open_positions': len(wallet['positions'])}
    learning['entry_policy'] = {'identity': stored['identity'], 'wallet': wallet, 'summary': summary}
    decisions = copy.deepcopy(signals)
    for signal in decisions.values():
        if signal['action'] not in ('BUY', 'HOLD'):
            continue
        if approved and 'shadow_model_score' in signal:
            signal['model_score'] = signal['shadow_model_score']
            signal['entry_allowed'] = True
            signal['action'] = ('BUY' if signal.get('feedback_entry_allowed', True)
                                and signal['model_score'] >= config['entry_threshold'] else 'HOLD')
        elif not baseline_entry_allowed:
            signal['entry_allowed'] = False
            signal['action'] = 'HOLD'
    return decisions, summary


def evaluate_online_policy(learning, signals, quotes, markets, errors, context_ready,
                           now, config, baseline_revision, baseline_entry_allowed):
    config = {key: value for key, value in config.items() if key != 'paper_entry_policy'}
    identity = {'config': copy.deepcopy(config), 'baseline_revision': baseline_revision,
                'evaluation_version': learning['evaluation_version'], 'search_version': 1}
    search = learning.get('threshold_search')
    if search is None or search['identity'] != identity:
        thresholds = sorted({config['entry_threshold'], *[value for value in (.45, .50, .55, .60)
                              if config['exit_threshold'] < value < config['entry_threshold']]})
        search = {'identity': identity, 'phase': 'selection',
                  'candidates': {str(value): None for value in thresholds},
                  'selected_threshold': None, 'confirmation': None}
    search.setdefault('failed_confirmations', [])
    search.setdefault('failed_selections', [])
    search.setdefault('selection_retry_count', 0)
    search.setdefault('last_selection_candle', -1)
    search.setdefault('retry_count', 0)
    search.setdefault('retry_reason', None)
    search.setdefault('retry_after', None)
    search.setdefault('evidence_after_candle', None)
    if search['phase'] == 'confirmation' and 'selected_candle' not in search:
        seen = [candle for candidate in search['candidates'].values() if candidate is not None
                for candle in candidate['wallet']['seen_candles'].values()]
        search['selected_candle'] = max(seen, default=(search['selected_at'] // INTERVAL_MS - 1) * INTERVAL_MS)
    observation = max((signal['candle_time'] for symbol, signal in signals.items()
                       if signal['action'] in ('BUY', 'HOLD') and 'shadow_model_score' in signal
                       and symbol in quotes and symbol in markets), default=None)
    healthy_observation = observation is not None and not errors and context_ready
    if (search['phase'] == 'retry_wait' and healthy_observation
            and now >= search['retry_after'] and observation > search['evidence_after_candle']):
        search.update(phase='selection', candidates={key: None for key in search['candidates']},
                      selected_threshold=None, confirmation=None, retry_after=None)
        search.pop('selected_at', None)
        search.pop('selected_candle', None)
        search.pop('last_confirmation_candle', None)
        search['last_selection_candle'] = -1
    candidates = {}
    confirmation_decisions = None
    if search['phase'] == 'selection':
        selection_signals = signals_after(signals, search['evidence_after_candle'])
        for key, stored in search['candidates'].items():
            trial = {**learning, 'entry_policy': stored}
            trial_config = {**config, 'entry_threshold': float(key)}
            _, summary = evaluate_fixed_policy(trial, selection_signals, quotes, markets, errors,
                                                context_ready, now, trial_config,
                                                baseline_revision, False)
            search['candidates'][key] = trial['entry_policy']
            candidates[key] = summary
        eligible = [(key, summary) for key, summary in candidates.items()
                    if summary['profitable_after_costs'] and not summary['halted']]
        if eligible and healthy_observation:
            selected, _ = max(eligible, key=lambda item: (item[1]['recent_net_pnl'], float(item[0])))
            search.update(phase='confirmation', selected_threshold=float(selected), selected_at=now,
                          selected_candle=observation)
        elif (healthy_observation and observation > search['last_selection_candle']
              and all(summary['halted'] for summary in candidates.values())
              and all(not candidate['wallet']['stale_positions']
                      for candidate in search['candidates'].values())):
            archive_exhausted_selection(search, observation, now)
        if healthy_observation:
            search['last_selection_candle'] = max(search['last_selection_candle'], observation)
    elif (search['phase'] == 'confirmation' and (search['confirmation'] is not None
          or healthy_observation and now > search['selected_at']
          and observation > search['selected_candle'])):
        trial = {**learning, 'entry_policy': search['confirmation']}
        trial_config = {**config, 'entry_threshold': search['selected_threshold']}
        confirmation_decisions, confirmation = evaluate_fixed_policy(
            trial, signals_after(signals, search['selected_candle']), quotes, markets, errors,
            context_ready, now, trial_config, baseline_revision, False)
        search['confirmation'] = trial['entry_policy']
        if (healthy_observation and not trial['entry_policy']['wallet']['stale_positions']
                and observation > search.get('last_confirmation_candle', search['selected_candle'])):
            search['last_confirmation_candle'] = observation
            reason = confirmation_failure(confirmation, search['selected_at'], now)
            if reason is not None:
                archive_failed_confirmation(search, reason, observation, now)
    candidates = {key: value['summary'] for key, value in search['candidates'].items()}
    confirmation = (search['confirmation'] or {}).get('summary')
    approved = bool(search['phase'] == 'confirmation' and confirmation and confirmation['approved'])
    summary = dict(confirmation or {
        'approved': False, 'better_than_baseline': active(learning['market'], 200),
        'profitable_after_costs': False, 'closed_trades': 0, 'recent_closed_trades': 0,
        'minimum_closed_trades': MIN_CLOSED_TRADES, 'recent_net_pnl': 0.,
        'realized_net_pnl': 0., 'total_net_pnl': 0., 'halted': False,
        'open_positions': 0, 'updated_at': now})
    summary.update(approved=approved, phase=search['phase'], selected_threshold=search['selected_threshold'],
                   retry_count=search['retry_count'], retry_reason=search['retry_reason'],
                   retry_after=search['retry_after'],
                   archived_confirmation_count=len(search['failed_confirmations']),
                   selection_retry_count=search['selection_retry_count'],
                   archived_selection_count=len(search['failed_selections']),
                   candidates=candidates, selection_closed_trades=sum(
                       item['closed_trades'] for item in candidates.values()),
                   selection_open_positions=sum(item['open_positions'] for item in candidates.values()))
    learning['threshold_search'] = search
    decisions = copy.deepcopy(signals)
    if approved:
        decisions = confirmation_decisions
    elif not baseline_entry_allowed:
        for signal in decisions.values():
            if signal['action'] in ('BUY', 'HOLD'):
                signal.update(action='HOLD', entry_allowed=False)
    return decisions, summary
