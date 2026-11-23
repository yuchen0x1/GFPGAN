import hashlib
import pytest
from io import BytesIO
from pathlib import Path

from gfpgan import model_registry
from gfpgan.model_registry import ModelSpec


class FakeResponse(BytesIO):

    def __init__(self, data, url):
        super().__init__(data)
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def make_spec(data=b'verified model bytes'):
    return ModelSpec(
        key='test',
        filename='test.pth',
        url='https://github.com/example/project/releases/download/v1/test.pth',
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
        family='test'), data


def test_registry_has_complete_integrity_metadata():
    assert {'1', '1.2', '1.3', '1.4', 'RestoreFormer'}.issubset(model_registry.MODEL_REGISTRY)
    for spec in model_registry.MODEL_REGISTRY.values():
        assert len(spec.sha256) == 64
        int(spec.sha256, 16)
        assert spec.url.startswith('https://')
        assert spec.size > 0


def test_download_is_verified_and_reused(monkeypatch, tmp_path):
    spec, data = make_spec()
    calls = []

    def fake_open(request, _timeout):
        calls.append(request.full_url)
        return FakeResponse(data, request.full_url)

    monkeypatch.setattr(model_registry, '_open_url', fake_open)
    path = model_registry.download_model(spec, tmp_path)
    assert path.read_bytes() == data
    assert model_registry.download_model(spec, tmp_path) == path
    assert calls == [spec.url]
    assert list(tmp_path.glob('*.part')) == []


def test_bad_hash_never_becomes_visible(monkeypatch, tmp_path):
    spec, _ = make_spec()
    monkeypatch.setattr(model_registry, '_open_url',
                        lambda request, _timeout: FakeResponse(b'tampered', request.full_url))
    with pytest.raises(RuntimeError, match='Failed to download verified model'):
        model_registry.download_model(spec, tmp_path, retries=1)
    assert not (tmp_path / spec.filename).exists()


def test_existing_corrupt_file_is_not_overwritten(tmp_path):
    spec, _ = make_spec()
    path = tmp_path / spec.filename
    path.write_bytes(b'keep for investigation')
    with pytest.raises(RuntimeError, match='size mismatch'):
        model_registry.download_model(spec, tmp_path)
    assert path.read_bytes() == b'keep for investigation'


def test_untrusted_or_insecure_urls_are_rejected(tmp_path):
    spec, data = make_spec()
    for url in ('http://github.com/model.pth', 'https://example.com/model.pth'):
        unsafe = ModelSpec(spec.key, spec.filename, url, hashlib.sha256(data).hexdigest(), len(data), spec.family)
        with pytest.raises(ValueError, match='untrusted URL'):
            model_registry.download_model(unsafe, tmp_path)


def test_pathlike_model_lookup():
    spec = model_registry.get_model_spec('1.4')
    assert model_registry.find_model_spec(Path(spec.filename)) == spec
