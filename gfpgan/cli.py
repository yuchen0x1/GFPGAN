"""Production-oriented command line interface for GFPGAN inference."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from gfpgan import GFPGANer
from gfpgan.model_registry import (MODEL_REGISTRY, default_model_dir, download_model, find_model_spec, get_model_spec,
                                   verify_model)

IMAGE_EXTENSIONS = frozenset({'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'})


def imread(path, flags=cv2.IMREAD_COLOR):
    """Read an image from Unicode paths on every supported platform."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except (OSError, ValueError):
        return cv2.imread(path, flags)


def imwrite(img, path, params=None):
    """Write an image to a Unicode path and create its parent directory."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or '.png'
    if ext in {'.jpg', '.jpeg'} and img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f'Failed to encode image for "{path}"')
    buf.tofile(path)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='gfpgan', description='Restore faces in a photo or an image directory with verified GFPGAN models.')
    parser.add_argument('-i', '--input', default='inputs/whole_imgs', help='Input image or directory.')
    parser.add_argument('-o', '--output', default='results', help='Output directory.')
    parser.add_argument('-v', '--version', choices=['1', '1.2', '1.3', '1.4', 'RestoreFormer'], default='1.3')
    parser.add_argument('-s', '--upscale', type=int, default=2, help='Final integer upscaling factor (default: 2).')
    parser.add_argument('--bg_upsampler', choices=['realesrgan', 'none'], default='realesrgan')
    parser.add_argument('--bg_tile', type=int, default=400, help='Background tile size; 0 disables tiling.')
    parser.add_argument('--suffix', default=None, help='Suffix added to restored filenames.')
    parser.add_argument('--only_center_face', action='store_true', help='Restore only the face nearest the image center.')
    parser.add_argument('--aligned', action='store_true', help='Treat each input as an already aligned face crop.')
    parser.add_argument('--ext', choices=['auto', 'jpg', 'jpeg', 'png', 'webp'], default='auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Reserved for API compatibility.')
    parser.add_argument('--model_path', help='Local GFPGAN checkpoint; overrides automatic model selection.')
    parser.add_argument('--models-dir', type=Path, default=default_model_dir(), help='Verified model cache directory.')
    parser.add_argument('--offline', action='store_true', help='Never access the network; require cached/local models.')
    parser.add_argument('--download-only', action='store_true', help='Download and verify required models, then exit.')
    parser.add_argument('--list-models', action='store_true', help='List pinned models and exit.')
    parser.add_argument('--fp32', action='store_true', help='Disable FP16 in the Real-ESRGAN background upsampler.')
    parser.add_argument('--device', help='Torch device such as cpu, cuda, cuda:1, or mps (default: automatic).')
    parser.add_argument('--no_parse', action='store_true', help='Skip face parsing during final blending.')
    parser.add_argument('--recursive', action='store_true', help='Process supported images in subdirectories.')
    parser.add_argument('--skip-existing', action='store_true', help='Do not overwrite an existing restored image.')
    parser.add_argument('--report', type=Path, help='Write a JSON batch report (for example results/report.json).')
    parser.add_argument('--strict', action='store_true', help='Return a non-zero exit code when any input fails.')
    return parser


def _list_models():
    print('key\tfamily\tfilename\tsha256')
    for spec in MODEL_REGISTRY.values():
        print(f'{spec.key}\t{spec.family}\t{spec.filename}\t{spec.sha256}')


def _discover_images(input_path, recursive):
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f'Unsupported input image extension: {input_path.suffix or "<none>"}')
        return [input_path], input_path.parent
    if not input_path.is_dir():
        raise FileNotFoundError(f'Input path does not exist: {input_path}')
    iterator = input_path.rglob('*') if recursive else input_path.glob('*')
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f'No supported images found in: {input_path}')
    return images, input_path


def _select_device(value):
    if value:
        device = torch.device(value)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device requested but CUDA is unavailable: {value}')
        if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise RuntimeError('MPS device requested but MPS is unavailable.')
        return device
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _resolve_face_model(args):
    if args.model_path:
        path = Path(args.model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Model file not found: {path}')
        known = find_model_spec(path)
        if known is not None:
            verify_model(path, known)
            return path, known.arch or 'clean', known.channel_multiplier or 2
        name = path.name
        if name == 'GFPGANv1.pth':
            return path, 'original', 1
        if 'RestoreFormer' in name:
            return path, 'RestoreFormer', 2
        return path, 'clean', 2

    spec = get_model_spec(args.version)
    legacy_locations = [
        Path('experiments/pretrained_models') / spec.filename,
        Path(__file__).resolve().parent / 'weights' / spec.filename,
    ]
    for path in legacy_locations:
        if path.is_file():
            verify_model(path, spec)
            return path.resolve(), spec.arch or 'clean', spec.channel_multiplier or 2
    path = download_model(spec, args.models_dir, offline=args.offline)
    return path, spec.arch or 'clean', spec.channel_multiplier or 2


def _build_background_upsampler(args, device):
    if args.bg_upsampler == 'none':
        return None
    if device.type == 'cpu':
        print('Background upscaling disabled on CPU because Real-ESRGAN is prohibitively slow; '
              'face restoration and Lanczos paste-back remain enabled.')
        return None
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    spec = get_model_spec('realesrgan-x2plus')
    model_path = download_model(spec, args.models_dir, offline=args.offline)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    return RealESRGANer(
        scale=2,
        model_path=str(model_path),
        model=model,
        tile=args.bg_tile,
        tile_pad=10,
        pre_pad=0,
        half=device.type == 'cuda' and not args.fp32,
        device=device)


def _relative_stem(path, root):
    try:
        return path.relative_to(root).with_suffix('')
    except ValueError:
        return Path(path.stem)


def _restored_output_path(args, relative_stem, original_suffix):
    extension = original_suffix.lstrip('.').lower() if args.ext == 'auto' else args.ext
    extension = extension or 'png'
    name = relative_stem.name + (f'_{args.suffix}' if args.suffix else '')
    return Path(args.output) / 'restored_imgs' / relative_stem.parent / f'{name}.{extension}'


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(str(temp), str(path))


def run(args):
    if args.list_models:
        _list_models()
        return 0
    if args.upscale < 1:
        raise ValueError('--upscale must be at least 1.')
    if args.bg_tile < 0:
        raise ValueError('--bg_tile cannot be negative.')

    started = time.monotonic()
    device = _select_device(args.device)
    model_path, arch, channel_multiplier = _resolve_face_model(args)
    # facexlib 0.3.0 constructs both helpers even when parsing is disabled.
    for key in ('retinaface-resnet50', 'parsenet'):
        download_model(get_model_spec(key), args.models_dir, offline=args.offline)
    if args.bg_upsampler == 'realesrgan' and device.type != 'cpu':
        download_model(get_model_spec('realesrgan-x2plus'), args.models_dir, offline=args.offline)
    if args.download_only:
        print(f'All required models are verified in {Path(args.models_dir).resolve()}')
        return 0

    images, input_root = _discover_images(Path(args.input).expanduser().resolve(), args.recursive)
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bg_upsampler = _build_background_upsampler(args, device)
    restorer = GFPGANer(
        model_path=str(model_path),
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler,
        device=device,
        use_parse=not args.no_parse,
        model_rootpath=args.models_dir,
        offline=args.offline)

    records = []
    failed = 0
    processed = 0
    skipped = 0
    for index, image_path in enumerate(images, 1):
        relative_stem = _relative_stem(image_path, input_root)
        restored_path = _restored_output_path(args, relative_stem, image_path.suffix)
        record = {'input': str(image_path), 'output': str(restored_path), 'status': 'pending', 'faces': 0}
        records.append(record)
        if args.skip_existing and restored_path.is_file():
            record['status'] = 'skipped-existing'
            skipped += 1
            print(f'[{index}/{len(images)}] Skipping existing {restored_path}')
            continue
        print(f'[{index}/{len(images)}] Processing {image_path.name} ...')
        try:
            input_img = imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if input_img is None:
                raise ValueError('OpenCV could not decode the image.')
            alpha = None
            if input_img.ndim == 2:
                input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
            elif input_img.ndim == 3 and input_img.shape[2] == 4:
                alpha = input_img[:, :, 3]
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR)

            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True,
                weight=args.weight)
            record['faces'] = len(restored_faces)
            if not cropped_faces:
                print(f'No faces detected in {image_path.name}; writing the upscaled/paste-back image.')

            for face_index, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                face_relative = relative_stem.parent / f'{relative_stem.name}_{face_index:02d}.png'
                imwrite(cropped_face, output_root / 'cropped_faces' / face_relative)
                face_name = face_relative.stem + (f'_{args.suffix}' if args.suffix else '') + '.png'
                imwrite(restored_face, output_root / 'restored_faces' / face_relative.parent / face_name)
                comparison = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(comparison, output_root / 'cmp' / face_relative)

            if restored_img is not None:
                if alpha is not None and restored_path.suffix.lower() == '.png':
                    resized_alpha = cv2.resize(
                        alpha, (restored_img.shape[1], restored_img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    restored_img = np.concatenate((restored_img[:, :, :3], resized_alpha[:, :, None]), axis=2)
                imwrite(restored_img, restored_path)
            record['status'] = 'processed'
            processed += 1
        except Exception as error:  # Continue the batch while preserving evidence in the report.
            record['status'] = 'failed'
            record['error'] = str(error)
            failed += 1
            print(f'Failed {image_path}: {error}')

    known_model = find_model_spec(model_path)
    report = {
        'schema_version': 1,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'model': args.version if not args.model_path else str(model_path),
        'model_sha256': known_model.sha256 if known_model else None,
        'device': str(device),
        'summary': {
            'total': len(images),
            'processed': processed,
            'skipped': skipped,
            'failed': failed,
            'duration_seconds': round(time.monotonic() - started, 3),
        },
        'items': records,
    }
    if args.report:
        report_path = args.report.expanduser().resolve()
        _write_report(report_path, report)
        print(f'Batch report: {report_path}')
    print(f'Completed: {processed} processed, {skipped} skipped, {failed} failed. Results: {output_root}')
    return 1 if failed and args.strict else 0


def main(argv=None):
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
