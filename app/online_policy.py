import copy

from app.online_learning import active
from app.portfolio import new_wallet, paper_cycle

MIN_CLOSED_TRADES = 20
EVIDENCE_MAX_AGE_MS = 7 * 86400000


def evaluate_online_policy(learning, signals, quotes, markets, errors, context_ready,
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
               'updated_at': now}
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
