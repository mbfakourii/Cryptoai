import copy

from app.online_learning import active
from app.portfolio import new_wallet, paper_cycle

MIN_CLOSED_TRADES = 20
EVIDENCE_MAX_AGE_MS = 7 * 86400000


def evaluate_fixed_policy(learning, signals, quotes, markets, errors, context_ready,
                           now, config, baseline_revision, baseline_approved):
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
        elif not baseline_approved:
            signal['entry_allowed'] = False
            signal['action'] = 'HOLD'
    return decisions, summary


def evaluate_online_policy(learning, signals, quotes, markets, errors, context_ready,
                           now, config, baseline_revision, baseline_approved):
    identity = {'config': copy.deepcopy(config), 'baseline_revision': baseline_revision,
                'evaluation_version': learning['evaluation_version'], 'search_version': 1}
    search = learning.get('threshold_search')
    if search is None or search['identity'] != identity:
        thresholds = sorted({config['entry_threshold'], *[value for value in (.45, .50, .55, .60)
                              if config['exit_threshold'] < value < config['entry_threshold']]})
        search = {'identity': identity, 'phase': 'selection',
                  'candidates': {str(value): None for value in thresholds},
                  'selected_threshold': None, 'confirmation': None}
    candidates = {}
    confirmation_decisions = None
    if search['phase'] == 'selection':
        for key, stored in search['candidates'].items():
            trial = {**learning, 'entry_policy': stored}
            trial_config = {**config, 'entry_threshold': float(key)}
            _, summary = evaluate_fixed_policy(trial, signals, quotes, markets, errors,
                                                context_ready, now, trial_config,
                                                baseline_revision, False)
            search['candidates'][key] = trial['entry_policy']
            candidates[key] = summary
        eligible = [(key, summary) for key, summary in candidates.items()
                    if summary['profitable_after_costs'] and not summary['halted']]
        if eligible and not errors and context_ready:
            selected, _ = max(eligible, key=lambda item: (item[1]['recent_net_pnl'], float(item[0])))
            search.update(phase='confirmation', selected_threshold=float(selected), selected_at=now)
    elif search['confirmation'] is not None or now > search['selected_at']:
        trial = {**learning, 'entry_policy': search['confirmation']}
        trial_config = {**config, 'entry_threshold': search['selected_threshold']}
        confirmation_decisions, confirmation = evaluate_fixed_policy(
            trial, signals, quotes, markets, errors, context_ready, now, trial_config,
            baseline_revision, False)
        search['confirmation'] = trial['entry_policy']
    candidates = {key: value['summary'] for key, value in search['candidates'].items()}
    confirmation = (search['confirmation'] or {}).get('summary')
    approved = bool(confirmation and confirmation['approved'])
    summary = dict(confirmation or {
        'approved': False, 'better_than_baseline': active(learning['market'], 200),
        'profitable_after_costs': False, 'closed_trades': 0, 'recent_closed_trades': 0,
        'minimum_closed_trades': MIN_CLOSED_TRADES, 'recent_net_pnl': 0.,
        'realized_net_pnl': 0., 'total_net_pnl': 0., 'halted': False,
        'open_positions': 0, 'updated_at': now})
    summary.update(phase=search['phase'], selected_threshold=search['selected_threshold'],
                   candidates=candidates, selection_closed_trades=sum(
                       item['closed_trades'] for item in candidates.values()),
                   selection_open_positions=sum(item['open_positions'] for item in candidates.values()))
    learning['threshold_search'] = search
    decisions = copy.deepcopy(signals)
    if approved:
        decisions = confirmation_decisions
    elif not baseline_approved:
        for signal in decisions.values():
            if signal['action'] in ('BUY', 'HOLD'):
                signal.update(action='HOLD', entry_allowed=False)
    return decisions, summary
