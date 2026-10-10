import json
import os
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

INTERVAL_SECONDS = 600
CYCLES_PER_RUN = 30
PORTFOLIO_WORKFLOW = 'paper-portfolio.yml'
SCHEDULER_WORKFLOW = 'portfolio-scheduler.yml'


class TransientDispatchError(RuntimeError):
    pass


def dispatch_workflow(repository, token, workflow, inputs=None):
    body = {'ref': 'main'}
    if inputs is not None:
        body['inputs'] = inputs
    request = Request(
        f'https://api.github.com/repos/{repository}/actions/workflows/{workflow}/dispatches',
        data=json.dumps(body).encode(),
        headers={'Authorization': f'Bearer {token}',
                 'Accept': 'application/vnd.github+json',
                 'Content-Type': 'application/json',
                 'X-GitHub-Api-Version': '2022-11-28'},
        method='POST')
    try:
        with urlopen(request, timeout=25) as response:
            if response.status not in (200, 204):
                raise RuntimeError(f'Workflow dispatch rejected: HTTP {response.status}.')
    except HTTPError as error:
        message = f'Workflow dispatch rejected: HTTP {error.code}.'
        if error.code == 429 or error.code >= 500:
            raise TransientDispatchError(message) from None
        raise RuntimeError(message) from None
    except (URLError, TimeoutError, OSError):
        raise TransientDispatchError('Workflow dispatch connection failed; outcome unknown.') from None


def run_scheduler(dispatch, cycles=CYCLES_PER_RUN, next_slot=None, clock=time.time, sleep=time.sleep):
    if not 1 <= cycles <= CYCLES_PER_RUN or (next_slot is not None and next_slot < 0):
        raise ValueError('Invalid scheduler window or next slot.')
    slot = int(clock() // INTERVAL_SECONDS) if next_slot is None else next_slot
    for _ in range(cycles):
        slot = max(slot, int(clock() // INTERVAL_SECONDS))
        while clock() < slot * INTERVAL_SECONDS:
            sleep(max(0, min(30, slot * INTERVAL_SECONDS - clock())))
        slot = max(slot, int(clock() // INTERVAL_SECONDS))
        timestamp = datetime.fromtimestamp(slot * INTERVAL_SECONDS, timezone.utc).isoformat()
        try:
            dispatch(PORTFOLIO_WORKFLOW)
            print(f'{timestamp}: portfolio cycle dispatched.', flush=True)
        except TransientDispatchError as error:
            print(f'{timestamp}: {error} Next slot will still run.', flush=True)
        slot += 1
    for attempt in range(3):
        try:
            dispatch(SCHEDULER_WORKFLOW, {'next_slot': str(slot), 'cycles': str(CYCLES_PER_RUN)})
            print(f'Scheduler continuation dispatched for slot {slot}.', flush=True)
            return
        except TransientDispatchError:
            if attempt == 2:
                raise
            sleep(5)


def main():
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise ValueError('The portfolio scheduler runs only in GitHub Actions.')
    repository, token = os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN')
    if not repository or not token:
        raise ValueError('GitHub Actions repository or token is missing.')
    cycles = int(os.environ.get('SCHEDULER_CYCLES') or CYCLES_PER_RUN)
    next_slot = os.environ.get('SCHEDULER_NEXT_SLOT')
    run_scheduler(lambda workflow, inputs=None: dispatch_workflow(repository, token, workflow, inputs),
                  cycles=cycles, next_slot=int(next_slot) if next_slot else None)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
