import argparse
import base64
import io
import subprocess
import sys
import zipfile

from app.core import MODEL_DIR
from app.signals import GitHubDeliveryState

FILES = ('BTCUSDT_xgboost.joblib', 'ETHUSDT_xgboost.joblib', 'runtime-requirements.txt')
MAX_BUNDLE_BYTES = 500_000


def unpack_bundle(content):
    if len(content) > MAX_BUNDLE_BYTES:
        raise ValueError('Model bundle exceeds the repository storage limit.')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if sorted(archive.namelist()) != sorted(FILES):
            raise ValueError('Unexpected files in model bundle.')
        if sum(item.file_size for item in archive.infolist()) > 2_000_000:
            raise ValueError('Expanded model bundle is too large.')
        files = {name: archive.read(name) for name in FILES}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (MODEL_DIR / name).write_bytes(payload)


def run_registry(command):
    state = GitHubDeliveryState(branch='cryptoai-models')
    path = '/contents/model-bundle.zip'
    existing = state.request('GET', path, params={'ref': state.branch})
    if command == 'download':
        if existing.status_code != 200:
            raise RuntimeError('No model bundle available. Run Train signal models first.')
        unpack_bundle(base64.b64decode(existing.json()['content']))
        print('Downloaded complete model bundle from cryptoai-models.')
        return
    if existing.status_code not in (200, 404):
        raise RuntimeError('Cannot read model registry; publication stopped.')
    versions = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / 'runtime-requirements.txt').write_bytes(versions)
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(MODEL_DIR / name, name)
    content = output.getvalue()
    if len(content) > MAX_BUNDLE_BYTES:
        raise ValueError('Model bundle exceeds 500 KB; publication stopped.')
    body = {'message': 'Publish complete trained model bundle', 'branch': state.branch,
            'content': base64.b64encode(content).decode()}
    if existing.status_code == 200:
        body['sha'] = existing.json()['sha']
    response = state.request('PUT', path, json=body)
    if response.status_code not in (200, 201):
        raise RuntimeError('Model publication failed; previous bundle remains active.')
    print(f'Published complete model bundle ({len(content)} bytes).')


def main():
    parser = argparse.ArgumentParser(description='Persist models in a repository branch without Actions artifacts.')
    parser.add_argument('command', choices=['upload', 'download'])
    args = parser.parse_args()
    try:
        run_registry(args.command)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
