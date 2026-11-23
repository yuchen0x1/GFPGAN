"""Pinned model metadata and verified, atomic downloads for GFPGAN."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    sha256: str
    size: int
    family: str
    arch: Optional[str] = None
    channel_multiplier: Optional[int] = None


_MODELS = {
    '1': ModelSpec(
        key='1',
        filename='GFPGANv1.pth',
        url='https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth',
        sha256='6db3a33dd00dd427b8a70a7e3c6244a5bcccb818736f4861ce1d609024a991de',
        size=615378983,
        family='face-restoration',
        arch='original',
        channel_multiplier=1),
    '1.2': ModelSpec(
        key='1.2',
        filename='GFPGANv1.2.pth',
        url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth',
        sha256='29e25ee90c170f4231163f4c83df6b59c968b73f3ce00cb884015ae005db083c',
        size=348632874,
        family='face-restoration',
        arch='clean',
        channel_multiplier=2),
    '1.3': ModelSpec(
        key='1.3',
        filename='GFPGANv1.3.pth',
        url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        sha256='c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70',
        size=348632874,
        family='face-restoration',
        arch='clean',
        channel_multiplier=2),
    '1.4': ModelSpec(
        key='1.4',
        filename='GFPGANv1.4.pth',
        url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
        sha256='e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad',
        size=348632874,
        family='face-restoration',
        arch='clean',
        channel_multiplier=2),
    'RestoreFormer': ModelSpec(
        key='RestoreFormer',
        filename='RestoreFormer.pth',
        url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
        sha256='07404d446d62ca3d5ed38b1de09a947a1e77d46dbccec961a74d713a8f24ace0',
        size=290785322,
        family='face-restoration',
        arch='RestoreFormer',
        channel_multiplier=2),
    'realesrgan-x2plus': ModelSpec(
        key='realesrgan-x2plus',
        filename='RealESRGAN_x2plus.pth',
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        sha256='49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb',
        size=67061725,
        family='background-upscaling'),
    'realesr-general-x4v3': ModelSpec(
        key='realesr-general-x4v3',
        filename='realesr-general-x4v3.pth',
        url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        sha256='8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292',
        size=4885111,
        family='background-upscaling'),
    'retinaface-resnet50': ModelSpec(
        key='retinaface-resnet50',
        filename='detection_Resnet50_Final.pth',
        url='https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        sha256='6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d',
        size=109497761,
        family='face-detection'),
    'parsenet': ModelSpec(
        key='parsenet',
        filename='parsing_parsenet.pth',
        url='https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
        sha256='3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2',
        size=85331193,
        family='face-parsing'),
}

MODEL_REGISTRY = MappingProxyType(_MODELS)

_ALLOWED_DOWNLOAD_HOSTS = frozenset({
    'github.com',
    'objects.githubusercontent.com',
    'release-assets.githubusercontent.com',
    'github-releases.githubusercontent.com',
})


def _sys_platform():
    import sys
    return sys.platform


def default_model_dir():
    configured = os.environ.get('GFPGAN_MODEL_DIR')
    if configured:
        return Path(configured).expanduser()
    if os.name == 'nt':
        base = Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local')))
        return base / 'GFPGAN' / 'models'
    if _sys_platform() == 'darwin':
        return Path.home() / 'Library' / 'Caches' / 'GFPGAN' / 'models'
    return Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'gfpgan' / 'models'


def get_model_spec(key):
    try:
        return MODEL_REGISTRY[key]
    except KeyError as error:
        choices = ', '.join(MODEL_REGISTRY)
        raise KeyError(f'Unknown model {key!r}. Available models: {choices}') from error


def find_model_spec(value):
    text = os.fspath(value)
    filename = Path(urlparse(text).path).name if '://' in text else Path(text).name
    for spec in MODEL_REGISTRY.values():
        if text == spec.url or filename == spec.filename:
            return spec
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path, spec):
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f'Model file not found: {candidate}')
    actual_size = candidate.stat().st_size
    if actual_size != spec.size:
        raise RuntimeError(
            f'Model size mismatch for {candidate}: expected {spec.size} bytes, got {actual_size}. '
            'Delete or move the file, then download it again.')
    actual_hash = sha256_file(candidate)
    if actual_hash.lower() != spec.sha256.lower():
        raise RuntimeError(
            f'Model SHA-256 mismatch for {candidate}: expected {spec.sha256}, got {actual_hash}. '
            'Do not use this file; delete or quarantine it and download it again.')
    return actual_hash


def _validate_download_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(f'Refusing model download from untrusted URL: {url}')


class _SafeRedirectHandler(HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_url(request, timeout):
    return build_opener(_SafeRedirectHandler()).open(request, timeout=timeout)


def download_model(spec, model_dir=None, *, offline=False, timeout=30, retries=3):
    """Return a verified local model, downloading atomically when necessary."""
    directory = Path(model_dir) if model_dir is not None else default_model_dir()
    directory = directory.expanduser().resolve()
    target = directory / spec.filename
    if target.exists():
        verify_model(target, spec)
        return target
    if offline:
        raise FileNotFoundError(
            f'Offline mode is enabled and the verified model is missing: {target}. '
            f'Download {spec.url} and verify SHA-256 {spec.sha256}.')

    _validate_download_url(spec.url)
    directory.mkdir(parents=True, exist_ok=True)
    request = Request(spec.url, headers={'User-Agent': 'GFPGAN-model-manager/1.0'})
    last_error = None  # type: Optional[Exception]

    for attempt in range(max(1, retries)):
        temp_path = None  # type: Optional[Path]
        try:
            with tempfile.NamedTemporaryFile(
                    mode='wb', prefix=f'.{spec.filename}.', suffix='.part', dir=str(directory), delete=False) as output:
                temp_path = Path(output.name)
                digest = hashlib.sha256()
                total = 0
                with _open_url(request, timeout) as response:
                    _validate_download_url(response.geturl())
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        output.write(chunk)
                        digest.update(chunk)
                        total += len(chunk)
                output.flush()
                os.fsync(output.fileno())

            actual_hash = digest.hexdigest()
            if total != spec.size:
                raise RuntimeError(f'expected {spec.size} bytes, received {total}')
            if actual_hash.lower() != spec.sha256.lower():
                raise RuntimeError(f'expected SHA-256 {spec.sha256}, received {actual_hash}')

            if target.exists():
                verify_model(target, spec)
                temp_path.unlink(missing_ok=True)
                return target
            os.replace(str(temp_path), str(target))
            return target
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as error:
            last_error = error
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if attempt + 1 < max(1, retries):
                time.sleep(2**attempt)

    raise RuntimeError(
        f'Failed to download verified model {spec.key!r} from {spec.url}: {last_error}. '
        f'You can download it manually to {target} and verify SHA-256 {spec.sha256}.') from last_error


def ensure_runtime_models(model_dir=None, *, use_parse=True, offline=False):
    keys = ['retinaface-resnet50']
    if use_parse:
        keys.append('parsenet')
    return {key: download_model(get_model_spec(key), model_dir, offline=offline) for key in keys}
