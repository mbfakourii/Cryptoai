import copy
import json

import pytest

from app.core import INTERVAL_MS
from app.online_learning import new_learning_state
from app.online_policy import EVIDENCE_MAX_AGE_MS, MAX_FAILED_CONFIRMATIONS, MAX_FAILED_SELECTIONS
from app.online_policy import evaluate_fixed_policy as evaluate_online_policy
from app.online_policy import evaluate_online_policy as evaluate_thresholds
from app.portfolio import load_config, new_wallet, paper_cycle


def setup_policy():
    learning = new_learning_state(4)
    learning['market'].update(samples=5000, evaluated=5000, evaluation_batches=96,
                              brier=.1, baseline_brier=.2)
    config = load_config()
    markets = {'AAAUSDT': {'symbol': 'AAAUSDT', 'step_size': '.001',
                           'min_qty': '.001', 'max_qty': '10000', 'min_notional': 5.}}
    return learning, config, markets


def test_changing_main_wallet_experiment_mode_preserves_shadow_evidence():
    learning, config, markets = setup_policy()
    config.pop('paper_entry_policy', None)
    cycle(learning, config, markets, 0, evaluator=evaluate_thresholds)
    cycle(learning, config, markets, 1, price=104., evaluator=evaluate_thresholds)
    before = copy.deepcopy(learning['threshold_search'])
    assert before['candidates']['0.65']['wallet']['closed_count'] == 1
    config['paper_entry_policy'] = 'after_training'
    cycle(learning, config, markets, 2, evaluator=evaluate_thresholds, baseline_approved=True)
    after = learning['threshold_search']
    assert after['identity'] == before['identity']
    assert after['candidates']['0.65']['wallet']['closed_count'] == 1


def cycle(learning, config, markets, candle, price=100., **overrides):
    signal = {'symbol': 'AAAUSDT', 'action': 'HOLD', 'model_score': .8,
              'shadow_model_score': .8, 'entry_allowed': False,
              'feedback_entry_allowed': True, 'atr': 1., 'horizon_bars': 4,
              'candle_time': candle * INTERVAL_MS}
    signal.update(overrides.pop('signal', {}))
    evaluator = overrides.pop('evaluator', evaluate_online_policy)
    return evaluator(
        learning, {'AAAUSDT': signal}, {'AAAUSDT': {'bid': price, 'ask': price + .01}},
        markets, overrides.pop('errors', {}), overrides.pop('ready', True),
        overrides.pop('now', (candle + 1) * INTERVAL_MS), config, overrides.pop('revision', 'baseline-v1'),
        overrides.pop('baseline_approved', False))


def profitable_history():
    learning, config, markets = setup_policy()
    for index in range(20):
        cycle(learning, config, markets, 2 * index, 100.)
        cycle(learning, config, markets, 2 * index + 1, 104.)
        learning = json.loads(json.dumps(learning))
    return learning, config, markets


def test_forward_shadow_profits_enable_actual_paper_entry_without_baseline_retraining():
    learning, config, markets = profitable_history()
    decisions, summary = cycle(learning, config, markets, 40)
    assert summary['approved'] and summary['recent_closed_trades'] == 20
    assert summary['recent_net_pnl'] > 0
    assert decisions['AAAUSDT']['action'] == 'BUY'
    wallet, events = paper_cycle(new_wallet(config), decisions,
                                 {'AAAUSDT': {'bid': 100., 'ask': 100.01}}, markets,
                                 {}, 41 * INTERVAL_MS, config)
    assert len(events) == 1 and events[0]['side'] == 'BUY'
    assert wallet['closed_count'] == 0


def test_one_profitable_trade_or_better_brier_alone_cannot_enable_entries():
    learning, config, markets = setup_policy()
    decisions, summary = cycle(learning, config, markets, 0)
    assert not summary['approved'] and decisions['AAAUSDT']['action'] == 'HOLD'
    _, summary = cycle(learning, config, markets, 1, 104.)
    assert summary['recent_net_pnl'] > 0 and not summary['approved']


def test_small_gross_gains_are_rejected_after_fees_spread_and_slippage():
    learning, config, markets = setup_policy()
    for index in range(20):
        cycle(learning, config, markets, index * 5, 100.)
        _, summary = cycle(learning, config, markets, index * 5 + 4, 100.1)
    assert summary['closed_trades'] == 20
    assert summary['realized_net_pnl'] < 0
    assert not summary['approved']


@pytest.mark.parametrize('failure', ['brier', 'candles', 'context', 'errors', 'halted'])
def test_approval_is_revoked_when_a_required_condition_fails(failure):
    learning, config, markets = profitable_history()
    kwargs = {}
    if failure == 'brier':
        learning['market']['brier'] = .3
    elif failure == 'candles':
        learning['market']['evaluation_batches'] = 95
    elif failure == 'context':
        kwargs['ready'] = False
    elif failure == 'errors':
        kwargs['errors'] = {'AAAUSDT': 'Unavailable'}
    else:
        learning['entry_policy']['wallet']['halted'] = True
    decisions, summary = cycle(learning, config, markets, 40, **kwargs)
    assert not summary['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'


def test_approval_does_not_override_trade_feedback_or_duplicate_candles():
    learning, config, markets = profitable_history()
    decisions, summary = cycle(learning, config, markets, 40,
                               signal={'feedback_entry_allowed': False})
    assert summary['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'
    decisions, _ = cycle(learning, config, markets, 41, signal={'action': 'WAITING'})
    assert decisions['AAAUSDT']['action'] == 'WAITING'


@pytest.mark.parametrize('change', ['revision', 'fees', 'threshold'])
def test_changed_policy_cannot_reuse_previous_profit_evidence(change):
    learning, config, markets = profitable_history()
    kwargs = {}
    if change == 'revision':
        kwargs['revision'] = 'baseline-v2'
    elif change == 'fees':
        config['fee_bps_per_side'] = 20
    else:
        config['entry_threshold'] = .7
    _, summary = cycle(learning, config, markets, 40, **kwargs)
    assert summary['closed_trades'] == 0 and not summary['approved']
    assert learning['market']['samples'] == 5000


def test_old_profits_expire_and_repeated_cycle_does_not_duplicate_shadow_trades():
    learning, config, markets = profitable_history()
    cycle(learning, config, markets, 40)
    original = copy.deepcopy(learning['entry_policy']['wallet'])
    cycle(learning, config, markets, 40)
    assert learning['entry_policy']['wallet'] == original
    _, summary = cycle(learning, config, markets, 40 + 8 * 96)
    assert summary['recent_closed_trades'] < 20 and not summary['approved']


def test_validated_baseline_entries_remain_available_during_shadow_warmup():
    learning, config, markets = setup_policy()
    decisions, summary = cycle(learning, config, markets, 0, baseline_approved=True,
                               signal={'action': 'BUY', 'entry_allowed': True})
    assert not summary['approved']
    assert decisions['AAAUSDT']['action'] == 'BUY'


def test_recent_losses_revoke_approval_even_with_positive_lifetime_profit():
    learning, config, markets = profitable_history()
    shadow = learning['entry_policy']['wallet']
    for event in shadow['trade_history']:
        if event['side'] == 'SELL':
            event['pnl'] = -.01
    decisions, summary = cycle(learning, config, markets, 40)
    assert summary['realized_net_pnl'] > 0
    assert summary['recent_net_pnl'] < 0
    assert not summary['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'


def test_approval_still_requires_the_current_signal_to_exceed_entry_threshold():
    learning, config, markets = profitable_history()
    decisions, summary = cycle(learning, config, markets, 40,
                               signal={'shadow_model_score': .5})
    assert summary['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'


def test_threshold_selection_requires_separate_future_profitable_confirmation():
    learning, config, markets = setup_policy()
    for index in range(40):
        decisions, summary = cycle(learning, config, markets, index * 2,
                                    evaluator=evaluate_thresholds,
                                    signal={'shadow_model_score': .5})
        assert not summary['approved']
        assert decisions['AAAUSDT']['action'] == 'HOLD'
        _, summary = cycle(learning, config, markets, index * 2 + 1, 104.,
                            evaluator=evaluate_thresholds,
                            signal={'shadow_model_score': .5})
        if index == 19:
            assert summary['phase'] == 'confirmation'
            assert summary['selected_threshold'] == .5
            assert summary['closed_trades'] == 0
            assert not summary['approved']
        if index < 39:
            assert not summary['approved']
        learning = json.loads(json.dumps(learning))
    assert summary['approved']
    assert summary['closed_trades'] == 20
    decisions, summary = cycle(learning, config, markets, 80,
                                evaluator=evaluate_thresholds,
                                signal={'shadow_model_score': .5})
    execution_config = {**config, 'entry_threshold': summary['selected_threshold']}
    _, events = paper_cycle(new_wallet(config), decisions,
                            {'AAAUSDT': {'bid': 100., 'ask': 100.01}}, markets, {},
                            81 * INTERVAL_MS, execution_config)
    assert len(events) == 1 and events[0]['side'] == 'BUY'
    assert summary['candidates']['0.65']['closed_trades'] == 0


def test_losing_threshold_candidates_do_not_get_selected_to_force_trades():
    learning, config, markets = setup_policy()
    for index in range(20):
        cycle(learning, config, markets, index * 5, evaluator=evaluate_thresholds)
        _, summary = cycle(learning, config, markets, index * 5 + 4, 100.1,
                            evaluator=evaluate_thresholds)
    assert summary['phase'] == 'selection'
    assert summary['selected_threshold'] is None
    assert not summary['approved']


def test_threshold_search_resumes_without_repeating_trades_and_resets_on_cost_change():
    learning, config, markets = setup_policy()
    cycle(learning, config, markets, 0, evaluator=evaluate_thresholds)
    original = copy.deepcopy(learning['threshold_search'])
    cycle(learning, config, markets, 0, evaluator=evaluate_thresholds)
    assert learning['threshold_search'] == original
    cycle(learning, config, markets, 1, 104., evaluator=evaluate_thresholds)
    config['slippage_bps_per_side'] = 10
    _, summary = cycle(learning, config, markets, 2, evaluator=evaluate_thresholds)
    assert summary['selection_closed_trades'] == 0
    assert learning['market']['samples'] == 5000


def select_profitable_threshold(learning, config, markets, first_candle=0):
    for index in range(20):
        cycle(learning, config, markets, first_candle + index * 2,
              evaluator=evaluate_thresholds)
        _, summary = cycle(learning, config, markets, first_candle + index * 2 + 1,
                            104., evaluator=evaluate_thresholds)
    assert summary['phase'] == 'confirmation'
    assert summary['closed_trades'] == 0
    return summary


def test_halted_confirmation_restarts_with_fresh_selection_and_preserves_failed_evidence():
    learning, config, markets = setup_policy()
    config['max_drawdown_fraction'] = .02
    learned = copy.deepcopy(learning['market'])
    main_wallet = new_wallet(config)
    main_wallet.update(cash=3., halted=True)
    learning['unrelated_main_wallet'] = copy.deepcopy(main_wallet)
    select_profitable_threshold(learning, config, markets)
    cycle(learning, config, markets, 40, evaluator=evaluate_thresholds)
    decisions, failed = cycle(learning, config, markets, 41, 90., evaluator=evaluate_thresholds)
    assert failed['phase'] == 'retry_wait'
    assert failed['retry_reason'] == 'risk_halt'
    assert failed['retry_count'] == 1 and not failed['approved']
    assert decisions['AAAUSDT']['action'] == 'HOLD'
    archive = copy.deepcopy(learning['threshold_search']['failed_confirmations'])
    assert archive[0]['summary']['halted']
    assert archive[0]['recent_closed_trades'][0]['pnl'] < 0
    original = copy.deepcopy(learning['threshold_search'])
    cycle(learning, config, markets, 41, 90., evaluator=evaluate_thresholds)
    assert learning['threshold_search'] == original
    _, repeated_candle = cycle(learning, config, markets, 42, evaluator=evaluate_thresholds,
                               signal={'candle_time': 41 * INTERVAL_MS})
    assert repeated_candle['phase'] == 'retry_wait'
    _, cooling_down = cycle(learning, config, markets, 42, evaluator=evaluate_thresholds,
                            now=42 * INTERVAL_MS + 60_000)
    assert cooling_down['phase'] == 'retry_wait'
    _, restarted = cycle(learning, config, markets, 42, evaluator=evaluate_thresholds)
    assert restarted['phase'] == 'selection'
    assert restarted['selection_closed_trades'] == 0
    assert restarted['closed_trades'] == 0 and not restarted['approved']
    for candidate in learning['threshold_search']['candidates'].values():
        assert candidate['wallet']['positions']['AAAUSDT']['signal_time'] == 42 * INTERVAL_MS
    for candle in range(43, 82):
        _, summary = cycle(learning, config, markets, candle,
                            104. if candle % 2 else 100., evaluator=evaluate_thresholds)
    assert summary['phase'] == 'confirmation'
    assert summary['closed_trades'] == 0 and not summary['approved']
    for candle in range(82, 122):
        _, summary = cycle(learning, config, markets, candle,
                            104. if candle % 2 else 100., evaluator=evaluate_thresholds)
    assert summary['approved'] and summary['closed_trades'] == 20
    assert learning['threshold_search']['failed_confirmations'] == archive
    assert learning['market'] == learned
    assert learning['unrelated_main_wallet'] == main_wallet


def test_confirmation_cannot_start_on_selection_candle_in_a_later_run():
    learning, config, markets = setup_policy()
    select_profitable_threshold(learning, config, markets)
    cycle(learning, config, markets, 40, evaluator=evaluate_thresholds,
          signal={'candle_time': 39 * INTERVAL_MS})
    assert learning['threshold_search']['confirmation'] is None
    cycle(learning, config, markets, 40, evaluator=evaluate_thresholds)
    assert learning['threshold_search']['confirmation']['wallet']['positions']


def test_confirmation_with_twenty_recent_losing_trades_retries_without_a_risk_halt():
    learning, config, markets = setup_policy()
    select_profitable_threshold(learning, config, markets)
    for index in range(20):
        cycle(learning, config, markets, 40 + index * 5, evaluator=evaluate_thresholds)
        _, summary = cycle(learning, config, markets, 44 + index * 5,
                            100.1, evaluator=evaluate_thresholds)
        if index < 19:
            assert summary['phase'] == 'confirmation'
    assert summary['phase'] == 'retry_wait'
    assert summary['retry_reason'] == 'unprofitable_confirmation'
    assert summary['closed_trades'] == 20 and not summary['halted']
    assert summary['recent_net_pnl'] < 0 and summary['total_net_pnl'] < 0


@pytest.mark.parametrize('temporary_failure', ['errors', 'context', 'brier'])
def test_temporary_unavailability_or_prediction_quality_does_not_restart_confirmation(temporary_failure):
    learning, config, markets = setup_policy()
    select_profitable_threshold(learning, config, markets)
    for candle in range(40, 80):
        _, previous = cycle(learning, config, markets, candle, 104. if candle % 2 else 100.,
                             evaluator=evaluate_thresholds)
    assert previous['approved']
    kwargs = {}
    if temporary_failure == 'errors':
        kwargs['errors'] = {'AAAUSDT': 'Unavailable'}
    elif temporary_failure == 'context':
        kwargs['ready'] = False
    else:
        learning['market']['brier'] = .3
    _, summary = cycle(learning, config, markets, 80, evaluator=evaluate_thresholds, **kwargs)
    assert summary['phase'] == 'confirmation'
    assert summary['closed_trades'] == previous['closed_trades'] == 20
    assert summary['retry_count'] == 0
    assert not learning['threshold_search']['failed_confirmations']


def test_independent_baseline_permission_is_preserved_during_failed_confirmation():
    learning, config, markets = setup_policy()
    config['max_drawdown_fraction'] = .02
    select_profitable_threshold(learning, config, markets)
    cycle(learning, config, markets, 40, evaluator=evaluate_thresholds)
    decisions, summary = cycle(learning, config, markets, 41, 90., evaluator=evaluate_thresholds,
                               baseline_approved=True, signal={'action': 'BUY', 'entry_allowed': True})
    assert summary['phase'] == 'retry_wait' and not summary['approved']
    assert decisions['AAAUSDT']['action'] == 'BUY'
    assert decisions['AAAUSDT']['entry_allowed']


def test_failed_confirmation_archive_is_bounded_across_retries():
    learning, config, markets = setup_policy()
    config['max_drawdown_fraction'] = .02
    for retry in range(MAX_FAILED_CONFIRMATIONS + 2):
        first = retry * 42
        select_profitable_threshold(learning, config, markets, first)
        cycle(learning, config, markets, first + 40, evaluator=evaluate_thresholds)
        _, summary = cycle(learning, config, markets, first + 41, 90., evaluator=evaluate_thresholds)
        learning = json.loads(json.dumps(learning))
    archives = learning['threshold_search']['failed_confirmations']
    assert summary['retry_count'] == MAX_FAILED_CONFIRMATIONS + 2
    assert len(archives) == summary['archived_confirmation_count'] == MAX_FAILED_CONFIRMATIONS
    assert archives[0]['failed_candle'] == (2 * 42 + 41) * INTERVAL_MS
    assert all(len(archive['recent_closed_trades']) <= 20 for archive in archives)


def test_legacy_confirmation_migrates_without_losing_its_evidence():
    learning, config, markets = setup_policy()
    select_profitable_threshold(learning, config, markets)
    cycle(learning, config, markets, 40, evaluator=evaluate_thresholds)
    cycle(learning, config, markets, 41, 104., evaluator=evaluate_thresholds)
    search = learning['threshold_search']
    original = copy.deepcopy(search['confirmation'])
    for key in ('failed_confirmations', 'retry_count', 'retry_reason', 'retry_after',
                'evidence_after_candle', 'selected_candle', 'last_confirmation_candle'):
        search.pop(key, None)
    _, summary = cycle(learning, config, markets, 41, 104., evaluator=evaluate_thresholds)
    assert search['confirmation'] == original
    assert summary['phase'] == 'confirmation' and summary['closed_trades'] == 1
    assert summary['retry_count'] == 0


def test_expired_failed_confirmation_is_archived_when_a_healthy_new_observation_arrives():
    learning, config, markets = setup_policy()
    select_profitable_threshold(learning, config, markets)
    legacy = copy.deepcopy(learning)
    for index in range(20):
        cycle(legacy, config, markets, 40 + index * 5)
        cycle(legacy, config, markets, 44 + index * 5, 100.1)
    learning['threshold_search']['confirmation'] = legacy['entry_policy']
    candle = 140 + EVIDENCE_MAX_AGE_MS // INTERVAL_MS
    _, unavailable = cycle(learning, config, markets, candle, evaluator=evaluate_thresholds,
                            errors={'AAAUSDT': 'Unavailable'})
    assert unavailable['phase'] == 'confirmation'
    _, expired = cycle(learning, config, markets, candle + 1, evaluator=evaluate_thresholds)
    assert expired['phase'] == 'retry_wait'
    assert expired['retry_reason'] == 'expired_unprofitable_confirmation'
    assert expired['recent_closed_trades'] == 0


def test_old_profitable_evidence_loses_approval_without_being_treated_as_a_failed_trial():
    learning, config, markets = setup_policy()
    select_profitable_threshold(learning, config, markets)
    for candle in range(40, 80):
        _, summary = cycle(learning, config, markets, candle, 104. if candle % 2 else 100.,
                            evaluator=evaluate_thresholds)
    assert summary['approved']
    _, expired = cycle(learning, config, markets, 80 + EVIDENCE_MAX_AGE_MS // INTERVAL_MS,
                        evaluator=evaluate_thresholds)
    assert not expired['approved']
    assert expired['phase'] == 'confirmation'
    assert expired['retry_count'] == 0 and expired['closed_trades'] == 20


def test_all_halted_selection_candidates_recover_using_only_future_observations():
    learning, config, markets = setup_policy()
    config['max_drawdown_fraction'] = .02
    learned = copy.deepcopy(learning['market'])
    main_wallet = new_wallet(config)
    main_wallet.update(cash=3., halted=True)
    learning['unrelated_main_wallet'] = copy.deepcopy(main_wallet)
    cycle(learning, config, markets, 0, evaluator=evaluate_thresholds)
    decisions, failed = cycle(learning, config, markets, 1, 90., evaluator=evaluate_thresholds)
    assert failed['phase'] == 'retry_wait'
    assert failed['retry_reason'] == 'selection_risk_halt'
    assert failed['retry_count'] == failed['selection_retry_count'] == 1
    assert failed['archived_confirmation_count'] == 0
    assert failed['archived_selection_count'] == 1
    assert not failed['approved'] and decisions['AAAUSDT']['action'] == 'HOLD'
    original = copy.deepcopy(learning['threshold_search'])
    for candidate in original['failed_selections'][0]['candidates'].values():
        assert candidate['summary']['halted']
        assert candidate['recent_closed_trades'][0]['pnl'] < 0
    cycle(learning, config, markets, 1, 90., evaluator=evaluate_thresholds)
    assert learning['threshold_search'] == original
    cycle(learning, config, markets, 2, evaluator=evaluate_thresholds,
          signal={'candle_time': INTERVAL_MS})
    assert learning['threshold_search'] == original
    cycle(learning, config, markets, 2, evaluator=evaluate_thresholds,
          now=2 * INTERVAL_MS + 60_000)
    assert learning['threshold_search'] == original
    _, restarted = cycle(learning, config, markets, 2, evaluator=evaluate_thresholds)
    assert restarted['phase'] == 'selection'
    assert restarted['selection_closed_trades'] == 0
    for candle in range(3, 82):
        _, summary = cycle(learning, config, markets, candle,
                            104. if candle % 2 else 100., evaluator=evaluate_thresholds)
    assert summary['approved'] and summary['closed_trades'] == 20
    assert learning['threshold_search']['failed_selections'] == original['failed_selections']
    assert learning['market'] == learned
    assert learning['unrelated_main_wallet'] == main_wallet


def test_unhalted_selection_candidates_keep_their_evidence_when_other_candidates_halt():
    learning, config, markets = setup_policy()
    config['max_drawdown_fraction'] = .02
    cycle(learning, config, markets, 0, evaluator=evaluate_thresholds,
          signal={'shadow_model_score': .5})
    _, summary = cycle(learning, config, markets, 1, 90., evaluator=evaluate_thresholds,
                        signal={'shadow_model_score': .5})
    assert summary['phase'] == 'selection'
    assert summary['candidates']['0.5']['halted']
    assert not summary['candidates']['0.65']['halted']
    assert summary['retry_count'] == 0
    assert learning['threshold_search']['candidates']['0.5']['wallet']['closed_count'] == 1


def test_exhausted_selection_waits_for_healthy_data_and_preserves_baseline_permission():
    learning, config, markets = setup_policy()
    config['max_drawdown_fraction'] = .02
    cycle(learning, config, markets, 0, evaluator=evaluate_thresholds)
    _, unavailable = cycle(learning, config, markets, 1, 90., evaluator=evaluate_thresholds,
                            errors={'AAAUSDT': 'Unavailable'})
    assert unavailable['phase'] == 'selection' and unavailable['retry_count'] == 0
    assert all(candidate['halted'] for candidate in unavailable['candidates'].values())
    decisions, failed = cycle(learning, config, markets, 2, evaluator=evaluate_thresholds,
                               baseline_approved=True, signal={'action': 'BUY', 'entry_allowed': True})
    assert failed['phase'] == 'retry_wait' and not failed['approved']
    assert decisions['AAAUSDT']['action'] == 'BUY' and decisions['AAAUSDT']['entry_allowed']


def test_failed_selection_archive_is_separate_and_bounded():
    learning, config, markets = setup_policy()
    config['max_drawdown_fraction'] = .02
    for retry in range(MAX_FAILED_SELECTIONS + 2):
        cycle(learning, config, markets, retry * 2, evaluator=evaluate_thresholds)
        _, summary = cycle(learning, config, markets, retry * 2 + 1, 90., evaluator=evaluate_thresholds)
        learning = json.loads(json.dumps(learning))
    archives = learning['threshold_search']['failed_selections']
    assert len(archives) == summary['archived_selection_count'] == MAX_FAILED_SELECTIONS
    assert summary['selection_retry_count'] == summary['retry_count'] == MAX_FAILED_SELECTIONS + 2
    assert summary['archived_confirmation_count'] == 0
    assert archives[0]['failed_candle'] == 5 * INTERVAL_MS
    assert all(sum(len(candidate['recent_closed_trades'])
                   for candidate in archive['candidates'].values()) <= 20 for archive in archives)
