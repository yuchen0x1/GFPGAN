# Changelog

## 1.4.0 (2026-11-17)

Fork release. In addition to the issue-by-issue triage of both upstream
repositories (TencentARC/GFPGAN and xinntao/Real-ESRGAN), this release
comprehensively upgrades the project:

### Compatibility layer (`gfpgan/_compat.py`, applied automatically)

- torchvision >= 0.17 shim: `torchvision.transforms.functional_tensor` is
  restored so the pinned basicsr keeps importing (#506, #510, #539, #556,
  #602, #603; Real-ESRGAN #768, #876, #765).
- Pure-PyTorch fallbacks for the StyleGAN2 CUDA ops: basicsr's
  `fused_act`/`upfirdn2d` only define their extension hooks when a compiled
  extension exists, which made the GFPGANv1 paper architecture crash with
  `NameError: name 'fused_act_ext' is not defined` (#608, #96, #115, #190)
  and unusable on CPU (#81, #384). The native implementations mirror the
  CUDA kernel's bias broadcast and flipped-kernel convolution exactly and
  are installed only when the compiled extension is absent.
- JPEG quality fix: basicsr passes a float quality to `cv2.imencode`,
  which OpenCV >= 4.8 rejects during training (`Bad argument ... 'imencode'`,
  #93, #222). The quality is cast to int.

### Runtime fixes

- `torch.load` uses `map_location='cpu'` and `weights_only=True` (torch 2.6+
  default) for GFPGAN weights, with clear errors for corrupted downloads
  (#303, #200); the training landmark files are loaded with the trusted
  loader they require on torch >= 2.6.
- Model weights directory is anchored to the package (absolute path) so
  facexlib no longer downloads into the current working directory (#442,
  #389, #174).
- `GFPGANer` accepts `device='cuda:1'` style strings and `use_parse=False`;
  `inference_gfpgan.py` exposes `--device`, `--no_parse`, `--fp32` and
  `--model_path` (#474, #148, #292, #414, #205).
- Unicode-safe image reading/writing for non-ASCII file names on Windows
  (#438, Real-ESRGAN #159); unreadable inputs are skipped with a warning;
  a warning is printed when no face is detected (#412).
- `-w/--weight` prints a warning instead of silently doing nothing: no
  released architecture consumes it (#423, #459).

### Packaging & tooling

- Modern PEP 621 packaging (`pyproject.toml`); version 1.4.0; Python
  3.8-3.12 classifiers; `setup.py` reduced to a legacy shim — use
  `pip install -e .`.
- numpy 2.x is supported and tested (`numpy>=1.20`).
- facexlib pinned to >= 0.3.0 (older releases lack `eye_dist_threshold`,
  #216, #168); `realesrgan` added to requirements so background
  enhancement works out of the box; stable `tensorboard` replaces
  `tb-nightly` (#418).
- Tag-triggered GitHub releases build and validate an sdist and wheel, then
  attach both distributions to the release without requiring a PyPI token.
- Test suite: the upstream arch tests now pass (they never ran without
  compiled extensions) plus new compatibility tests; `pytest` works from a
  plain checkout (`tests/conftest.py`).
- CI: lint workflow modernized and a new CPU test workflow added
  (Python 3.11/3.12).

- Dockerfile for CPU inference, buildable from the repository root.

### Verification

All of the above is covered by `pytest` (18 tests) and verified on
Python 3.11 + torch 2.7.1+cu128 (RTX 5060, GPU) and CPU-only execution,
with numpy 1.26 and numpy 2.4.
