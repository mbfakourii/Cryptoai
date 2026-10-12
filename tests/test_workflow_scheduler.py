import io
from urllib.error import HTTPError, URLError

import pytest

from app import workflow_scheduler as scheduler


class Clock:
    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_dispatch_duration_does_not_shift_fifteen_minute_boundaries():
    clock, calls = Clock(125), []

    def dispatch(workflow, inputs=None):
        calls.append((workflow, clock.now, inputs))
        clock.now += 35

    scheduler.run_scheduler(dispatch, cycles=3, clock=clock.time, sleep=clock.sleep)
    assert [call[1] for call in calls[:-1]] == [125, 900, 1800]
    assert calls[-1][0] == scheduler.SCHEDULER_WORKFLOW
    assert calls[-1][2] == {'next_candle_slot': '3', 'cycles': '20'}


def test_handoff_waits_for_next_slot_without_dispatching_it_early():
    clock, calls = Clock(125), []
    scheduler.run_scheduler(lambda workflow, inputs=None: calls.append((workflow, clock.now, inputs)),
                            cycles=1, next_slot=1, clock=clock.time, sleep=clock.sleep)
    assert calls[0] == (scheduler.PORTFOLIO_WORKFLOW, 900, None)
    assert calls[1][2]['next_candle_slot'] == '2'


def test_delayed_runner_skips_missed_slots_instead_of_replaying_them():
    clock, calls = Clock(1950), []
    scheduler.run_scheduler(lambda workflow, inputs=None: calls.append((workflow, clock.now, inputs)),
                            cycles=1, next_slot=1, clock=clock.time, sleep=clock.sleep)
    assert calls[0][1] == 1950
    assert calls[1][2]['next_candle_slot'] == '3'


def test_temporary_dispatch_failure_does_not_stop_later_cycles():
    clock, calls = Clock(0), []

    def dispatch(workflow, inputs=None):
        calls.append((workflow, clock.now, inputs))
        if len(calls) == 1:
            raise scheduler.TransientDispatchError('Temporary failure')

    scheduler.run_scheduler(dispatch, cycles=2, clock=clock.time, sleep=clock.sleep)
    assert calls[1][:2] == (scheduler.PORTFOLIO_WORKFLOW, 900)
    assert calls[2][0] == scheduler.SCHEDULER_WORKFLOW


def test_handoff_retries_preserve_the_next_slot():
    clock, calls = Clock(0), []

    def dispatch(workflow, inputs=None):
        calls.append((workflow, inputs))
        if workflow == scheduler.SCHEDULER_WORKFLOW and len(calls) < 4:
            raise scheduler.TransientDispatchError('Temporary failure')

    scheduler.run_scheduler(dispatch, cycles=1, clock=clock.time, sleep=clock.sleep)
    assert [call[1] for call in calls[1:]] == [{'next_candle_slot': '1', 'cycles': '20'}] * 3


def test_permission_failure_stops_timer_without_spawning_another_timer():
    calls = []

    def dispatch(workflow, inputs=None):
        calls.append(workflow)
        raise RuntimeError('HTTP 403')

    with pytest.raises(RuntimeError, match='403'):
        scheduler.run_scheduler(dispatch, cycles=1)
    assert calls == [scheduler.PORTFOLIO_WORKFLOW]


@pytest.mark.parametrize('cycles,next_slot', [(0, None), (21, None), (1, -1)])
def test_invalid_windows_do_not_dispatch(cycles, next_slot):
    with pytest.raises(ValueError):
        scheduler.run_scheduler(lambda *args: pytest.fail('Unexpected dispatch'),
                                cycles=cycles, next_slot=next_slot)


def test_dispatch_targets_main_with_authenticated_json(monkeypatch):
    requests = []

    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def send(request, timeout):
        requests.append(request)
        return Response()

    monkeypatch.setattr(scheduler, 'urlopen', send)
    scheduler.dispatch_workflow('owner/repo', 'test-token', scheduler.PORTFOLIO_WORKFLOW)
    request = requests[0]
    assert request.full_url.endswith('/paper-portfolio.yml/dispatches')
    assert request.method == 'POST'
    assert request.data == b'{"ref": "main"}'
    assert request.get_header('Authorization') == 'Bearer test-token'


@pytest.mark.parametrize('code,exception', [(403, RuntimeError), (429, scheduler.TransientDispatchError),
                                         (503, scheduler.TransientDispatchError)])
def test_http_failures_never_expose_response_secrets(monkeypatch, code, exception):
    def fail(*args, **kwargs):
        raise HTTPError('https://api.github.com', code, 'test-token', {}, io.BytesIO(b'test-token'))

    monkeypatch.setattr(scheduler, 'urlopen', fail)
    with pytest.raises(exception) as error:
        scheduler.dispatch_workflow('owner/repo', 'test-token', scheduler.PORTFOLIO_WORKFLOW)
    assert 'test-token' not in str(error.value)
    assert str(code) in str(error.value)


def test_network_failures_never_expose_credentials(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError('test-token')

    monkeypatch.setattr(scheduler, 'urlopen', fail)
    with pytest.raises(scheduler.TransientDispatchError) as error:
        scheduler.dispatch_workflow('owner/repo', 'test-token', scheduler.PORTFOLIO_WORKFLOW)
    assert 'test-token' not in str(error.value)


def test_scheduler_cannot_run_on_a_local_machine(monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    with pytest.raises(ValueError, match='only in GitHub Actions'):
        scheduler.main()


def test_legacy_timer_handoff_does_not_reinterpret_ten_minute_slots(monkeypatch):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_TOKEN', 'test-token')
    monkeypatch.setenv('SCHEDULER_CYCLES', '30')
    monkeypatch.setenv('SCHEDULER_NEXT_SLOT', '1234567')
    monkeypatch.delenv('SCHEDULER_NEXT_CANDLE_SLOT', raising=False)
    calls = []
    monkeypatch.setattr(scheduler, 'run_scheduler', lambda dispatch, **kwargs: calls.append(kwargs))
    scheduler.main()
    assert calls == [{'cycles': 20, 'next_slot': None}]
