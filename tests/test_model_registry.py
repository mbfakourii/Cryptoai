import io
import zipfile

import pytest

from app import model_registry


def bundle(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return output.getvalue()


def test_complete_model_pair_and_runtime_are_restored_together(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, 'MODEL_DIR', tmp_path)
    files = {name: name.encode() for name in model_registry.FILES}
    model_registry.unpack_bundle(bundle(files))
    assert {name: (tmp_path / name).read_bytes() for name in files} == files


def test_unexpected_paths_cannot_escape_model_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, 'MODEL_DIR', tmp_path / 'models')
    files = {name: b'data' for name in model_registry.FILES}
    files['../outside'] = b'bad'
    with pytest.raises(ValueError):
        model_registry.unpack_bundle(bundle(files))
    assert not (tmp_path / 'models').exists()


def test_partial_model_pair_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, 'MODEL_DIR', tmp_path)
    with pytest.raises(ValueError):
        model_registry.unpack_bundle(bundle({'BTCUSDT_xgboost.joblib': b'data'}))
    assert not list(tmp_path.iterdir())


def test_highly_compressed_oversized_bundle_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, 'MODEL_DIR', tmp_path)
    files = {name: b'0' * 800_000 for name in model_registry.FILES}
    with pytest.raises(ValueError):
        model_registry.unpack_bundle(bundle(files))
    assert not list(tmp_path.iterdir())
