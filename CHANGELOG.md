# Changelog

## 1.4.1 (2026-11-23)

Patch release focused on reproducible inference and release quality:

- Fixed the CI lint failures in the CLI and model registry.
- Accepted `pathlib.Path` checkpoints through the public `GFPGANer` API.
- Added `--seed` to seed Python, NumPy, and PyTorch inference RNGs; batch reports
  now record the seed and effective restoration options.
- Fixed validation image writes on Windows paths containing non-ASCII characters.

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

### Full-codebase review fixes (2026-11-18)

- Training is repaired on modern PyTorch: `setup_optimizers` passed
  `betas = (0**1, 0.99**1)` to Adam, whose first element is the *int* `0`;
  torch >= 2.6 rejects betas that are not consistently floats, so
  constructing `GFPGANModel` (i.e. any training run) failed immediately.
- `get_roi_regions` now derives the ground-truth eye/mouth regions from
  `self.gt.detach()`: gt gradients are never used, and when gt carries
  `requires_grad` (left on by the R1 regularization of a previous step —
  any scenario that reuses the gt tensor across steps) the left and right
  eye regions shared one roi_align subgraph whose freeing made the second
  facial-D backward raise "Trying to backward through the graph a second
  time".
- `inference_gfpgan.py` no longer skips writing the restored whole image
  when no face is detected: with a background upsampler (or the default
  Lanczos paste-back) the upscaled image is still the expected output; the
  "no faces" warning stays (#412).
- `GFPGANBilinear.forward` accepts and ignores extra kwargs (e.g. the
  `weight` `GFPGANer.enhance` always passes), so `arch='bilinear'` no
  longer raises `TypeError`.
- cog predictor: RGBA inputs are converted to BGR for detection/restoration
  and the alpha channel is re-attached to the saved output; previously a
  4-channel image made the networks raise and the run fell back to the
  unprocessed input.
- `tests/conftest.py` imports `gfpgan` before any test module so test files
  that `import basicsr` directly (test_gfpgan_model) collect on
  torchvision >= 0.17; `test_gfpgan_model` falls back to `num_gpu: 0` when
  no CUDA device is present (now runnable on CPU); `test_utils` skips when
  the pretrained models are not downloaded; the fused_leaky_relu shim
  comment now documents the true broadcast contract (per-last-dim for the
  2D/3D `EqualLinear` inputs, per-dim-1 channel for 4D conv inputs).

### Verification

All of the above is covered by `pytest` (20 tests) and verified on
Python 3.11 + torch 2.7.1+cu128 (RTX 5060, GPU) and CPU-only execution,
with numpy 1.26 and numpy 2.4.
