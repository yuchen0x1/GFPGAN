<p align="center">
  <img src="assets/gfpgan_logo.png" height=130>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">
<!-- <a href="https://twitter.com/_Xintao_" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/17445847/187162058-c764ced6-952f-404b-ac85-ba95cce18e7b.png" width="4%" alt="" />
</a> -->

[![Tests](https://github.com/yuchen0x1/GFPGAN/actions/workflows/tests.yml/badge.svg)](https://github.com/yuchen0x1/GFPGAN/actions/workflows/tests.yml)
[![Lint](https://github.com/yuchen0x1/GFPGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/yuchen0x1/GFPGAN/actions/workflows/pylint.yml)
[![Release](https://img.shields.io/github/v/release/yuchen0x1/GFPGAN?sort=semver)](https://github.com/yuchen0x1/GFPGAN/releases)
[![LICENSE](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)
</div>

## :fork_and_knife: About this fork

This is a maintained fork of [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) (based on upstream v1.3.8), with its package version tracked in [pyproject.toml](pyproject.toml). Highlights:

- **Runs everywhere**: modern environment compatibility — torchvision >= 0.17, torch >= 2.6/2.7 (incl. RTX 50xx/Blackwell), numpy 2.x, Python 3.8–3.12.
- **The paper model (V1) works again**: the StyleGAN2 CUDA ops now get exact pure-PyTorch fallbacks, so GFPGANv1 runs without compiled CUDA extensions — including on CPU — with no `BASICSR_JIT`/`BASICSR_EXT` builds.
- **~60 reproducible upstream issues fixed**, found by triaging every open issue of GFPGAN and Real-ESRGAN (961 items): see [CHANGELOG.md](CHANGELOG.md) and the [Troubleshooting](#ambulance-troubleshooting--compatibility) section.
- **Quality infrastructure**: 18-test suite (green in CI on Python 3.11/3.12), lint workflow, CPU Docker image, modern PEP 621 packaging.
- **New CLI options**: `--device`, `--fp32`, `--model_path`, `--no_parse`, `--seed`; `GFPGANer(..., device=..., use_parse=...)`.

The rest of this document is the upstream README, updated where this fork differs.

> This is a community-maintained fork of GFPGAN. It is not an official TencentARC release.

## Latest release

<!-- github-release:start -->

**[v1.4.2](https://github.com/yuchen0x1/GFPGAN/releases/tag/v1.4.2)** · 2026-11-26

本次更新主要改进了 Docker 镜像构建与本地开发钩子配置。修复了 Docker 构建必然失败的问题，并修复了 pre-commit 钩子版本过旧导致的本地钩子不可用问题。本次为维护性补丁版本，不涉及任何推理或训练行为的变化，可直接覆盖升级。

[Release history](https://github.com/yuchen0x1/GFPGAN/releases)

<!-- github-release:end -->

[Upstream model weights](https://github.com/TencentARC/GFPGAN/releases)

### What is included in this fork

- A compatibility shim for modern torchvision and BasicSR, including a pure-PyTorch fallback for the StyleGAN2 operators used by the original V1 model.
- CPU-safe checkpoint loading and inference, with explicit device selection for `cpu`, `cuda`, and `cuda:N`.
- Unicode-safe image I/O on Windows, clearer errors for missing or corrupted weights, and a package-anchored directory for facexlib models.
- Reproducible CPU tests and lint checks that run in GitHub Actions on Python 3.11 and 3.12.

The fork keeps the original model weights, inference API, training entry points, and Apache 2.0 license. See [CHANGELOG.md](CHANGELOG.md) for the full list of changes.

1. :boom: **Updated** online demo: [![Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/tencentarc/gfpgan). Here is the [backup](https://replicate.com/xinntao/gfpgan).
1. :boom: **Updated** online demo: [![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/Xintao/GFPGAN)
1. [Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo) for GFPGAN <a href="https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>; (Another [Colab Demo](https://colab.research.google.com/drive/1Oa1WwKB4M4l1GmR7CtswDVgOCOeSLChA?usp=sharing) for the original paper model)

<!-- 3. Online demo: [Replicate.ai](https://replicate.com/xinntao/gfpgan) (may need to sign in, return the whole image)
4. Online demo: [Baseten.co](https://app.baseten.co/applications/Q04Lz0d/operator_views/8qZG6Bg) (backed by GPU, returns the whole image)
5. We provide a *clean* version of GFPGAN, which can run without CUDA extensions. So that it can run in **Windows** or on **CPU mode**. -->

> :rocket: **Thanks for your interest in our work. You may also want to check our new updates on the *tiny models* for *anime images and videos* in [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_video_model.md)** :blush:

GFPGAN aims at developing a **Practical Algorithm for Real-world Face Restoration**.<br>
It leverages rich and diverse priors encapsulated in a pretrained face GAN (*e.g.*, StyleGAN2) for blind face restoration.

:question: Frequently Asked Questions can be found in [FAQ.md](FAQ.md).

:triangular_flag_on_post: **Updates**

- :white_check_mark: Add [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) inference codes.
- :white_check_mark: Add [V1.4 model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth), which produces slightly more details and better identity than V1.3.
- :white_check_mark: Add **[V1.3 model](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)**, which produces **more natural** restoration results, and better results on *very low-quality* / *high-quality* inputs. See more in [Model zoo](#european_castle-model-zoo), [Comparisons.md](Comparisons.md)
- :white_check_mark: Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See [Gradio Web Demo](https://huggingface.co/spaces/akhaliq/GFPGAN).
- :white_check_mark: Support enhancing non-face regions (background) with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
- :white_check_mark: We provide a *clean* version of GFPGAN, which does not require CUDA extensions.
- :white_check_mark: We provide an updated model without colorizing faces.

---

If GFPGAN is helpful in your photos/projects, please help to :star: this repo or recommend it to your friends. Thanks:blush:
Other recommended projects:<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image restoration<br>
:arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR): An open-source image and video restoration toolbox<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): A collection that provides useful face-relation functions<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): A PyQt5-based image viewer that is handy for view and comparison<br>

---

### :book: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior

> [[Paper](https://arxiv.org/abs/2101.04061)] &emsp; [[Project Page](https://xinntao.github.io/projects/gfpgan)] &emsp; [Demo] <br>
> [Xintao Wang](https://xinntao.github.io/), [Yu Li](https://yu-li.github.io/), [Honglun Zhang](https://scholar.google.com/citations?hl=en&user=KjQLROoAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG

<p align="center">
  <img src="https://xinntao.github.io/projects/GFPGAN_src/gfpgan_teaser.jpg">
</p>

---

## :wrench: Dependencies and Installation

- Python >= 3.8 (3.11/3.12 recommended; the test suite runs on both. Python 3.13 cannot
  install basicsr 1.4.2 yet: it ships an sdist whose setup.py predates the Python 3.13
  exec-locals change — use 3.11/3.12 until basicsr releases a fixed version)
- [PyTorch >= 1.7](https://pytorch.org/) (RTX 50xx GPUs need torch >= 2.7 with CUDA 12.8 wheels)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- numpy 1.20+ (numpy 2.x supported and tested)

### Installation

GFPGAN ships a *clean* version that does not require customized CUDA
extensions, and this fork additionally makes the original paper model (V1)
run without compiled CUDA extensions and on CPU — no `BASICSR_JIT` /
`BASICSR_EXT` builds needed for either.

1. Clone repo

    ```bash
    git clone https://github.com/yuchen0x1/GFPGAN.git
    cd GFPGAN
    ```

1. Install dependent packages

    ```bash
    # basicsr still declares tb-nightly, which is unavailable on some Python
    # versions: install its real runtime deps manually, then install the
    # basicsr/facexlib/realesrgan packages without resolving deps.
    pip install addict future lmdb opencv-python pyyaml scipy tensorboard tqdm yapf requests scikit-image
    pip install basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0 --no-deps

    pip install -r requirements.txt
    pip install -e .
    ```

### GitHub releases

This fork does not publish the upstream-occupied `gfpgan` package name to
PyPI. For maintainers, pushing a `vX.Y.Z` tag whose version matches both
`pyproject.toml` and `VERSION` starts [`.github/workflows/release.yml`](.github/workflows/release.yml).
The workflow builds and validates a source distribution and wheel, stores them
as workflow artifacts, and attaches them to a GitHub Release.

1. Run the tests (optional but recommended)

    ```bash
    pip install pytest
    pytest tests/
    ```

    The full suite runs on CPU and GPU alike; `tests/test_utils.py` skips
    itself when the pretrained models are not present under
    `experiments/pretrained_models`.

## :ambulance: Troubleshooting & Compatibility

This fork keeps GFPGAN usable on modern environments. The following upstream issues are addressed:

- **`ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`** (#506, #510, #539, #556, #602, #603): importing `gfpgan` now installs a compatibility shim automatically, so the pinned basicsr keeps working with torchvision >= 0.17. If you import `basicsr` or `realesrgan` directly, do `import gfpgan` first.
- **`No matching distribution found for tb-nightly`** (#418): `requirements.txt` now depends on the stable `tensorboard`. Note that basicsr itself still declares `tb-nightly`; if `pip install basicsr` fails on your Python version, install the other requirements first and then `pip install basicsr==1.4.2 --no-deps`.
- **NumPy issues** (#182, #224): the old errors came from torch builds compiled against NumPy 1.x running under NumPy 2.x. With torch >= 2.7 (and everything this fork patches), numpy 2.x is supported and tested; the requirement is `numpy>=1.20`.
- **GPU checkpoints fail to load on CPU-only machines**: model files are loaded with `map_location='cpu'` and the safe `weights_only=True` loader (also required by torch >= 2.6).
- **Corrupted / invalid model files** (#303, #200): errors such as `PytorchStreamReader failed reading zip archive` or `argument of type 'NoneType' is not iterable` now raise a clear message telling you to re-download the model.
- **Model weights downloaded to the wrong folder** (#442, #389, #174): the facexlib detection/parsing weights directory is now anchored to the package (absolute path), so GFPGAN works no matter which directory you launch it from.
- **Choosing the device** (#474, #148): pass `GFPGANer(..., device='cuda:1')` or use the new `--device` flag (`cuda`, `cuda:1`, `cpu`) in `inference_gfpgan.py`.
- **Non-ASCII file names fail on Windows** (#438, Real-ESRGAN #159): image reading/writing now goes through Unicode-safe helpers.
- **Old facexlib versions** (#216, #168): `get_face_landmarks_5() got an unexpected keyword argument 'eye_dist_threshold'` means your facexlib is too old; `requirements.txt` now requires `facexlib>=0.3.0`.
- **Black background with enhanced face** (#205, #255, Real-ESRGAN #116): the half-precision background upsampler fails on some GPUs; pass the new `--fp32` flag to `inference_gfpgan.py` to disable it.
- **Offline / blocked GitHub downloads** (#292, #414): pass `--model_path /path/to/GFPGANv1.4.pth` to use a local model file; face detection/parsing weights go to `gfpgan/weights/`.
- **No output image although the run "succeeds"** (#412): usually means no face was detected; the script now prints `No faces detected ...` instead of writing nothing.
- **`-w/--weight` does not change results**: no released architecture consumes this option (every `forward` swallows it via `**kwargs`). The flag now prints a warning instead of silently doing nothing; it is kept for API compatibility.
- **Paper model V1 import/forward errors** (`deform_conv.so`, `fused_act_ext`, `BASICSR_JIT` hangs) (#81, #125, #135, #191, #212, #558, #608, #96): **fixed in this fork**. The StyleGAN2 CUDA ops now get exact pure-PyTorch fallbacks when no compiled extension is present, so the V1 paper architecture runs everywhere — including on CPU — with no `BASICSR_JIT`/`BASICSR_EXT` builds. The clean models (V1.2–V1.4) remain the recommended defaults.
- **Custom-trained checkpoints** (#65, #411, #464, #489, #550): load your `net_g_*.pth` with `--model_path` (or `GFPGANer(model_path=...)`); state-dict key mismatches mean the file was saved with a different `arch` / `channel_multiplier`.
- **GPU utilization stays low** (#370, #415, #477, #479, #634): expected — face tensors are small and pre/post-processing runs on CPU. Throughput scales with the number of images, not with larger `--upscale`.
- **Face blending artifacts / square blocks** (#294, #316, #359, #590): caused by the facexlib paste-back mask and the fixed 512 face crop (hair/edges outside the box are untouched). Try `--fp32`; a larger crop box is a known limitation, not a bug.
- **Video / batch processing** (#164, #334, #407, #552): split the video into frames and call the script per frame, or loop over `GFPGANer.enhance()`; there is no batched single-call API.
- **Multiprocessing crashes** (#574): use the `'spawn'` start method when combining CUDA with `multiprocessing`.
- **RTX 50xx (Blackwell) GPUs** ("no kernel image" errors): your PyTorch is too old. Use torch >= 2.7 with the CUDA 12.8 wheels (`pip install torch --index-url https://download.pytorch.org/whl/cu128`) on Python >= 3.10.
- **Training failures** (#93, #215, #545, #564): training requires exactly `basicsr==1.4.2`; `ModuleNotFoundError: No module named 'basicsr.losses.gan_loss'` means an older basicsr is installed.
- **Docker** (#102, #194, #306): a maintained community image is available at [docker-gfpgan](https://github.com/soulteary/docker-gfpgan).

Known limitations of the original project (not fixable in code):
- AMD / Intel ARC / Apple-Silicon GPUs are not supported by PyTorch CUDA builds (#203, #417, #547, #643); those machines fall back to CPU.
- Background enhancement is skipped in CPU mode (#109, #445), so only faces are restored there (#280, #299, #429).
- FP16 inference and ONNX / TensorRT / TorchScript export are not officially supported (#50, #104, #238, #275, #480, #483, #587).
- The restored identity may differ slightly from the input, and the model smooths skin / alters hair style on some photos (#139, #159, #167, #171, #180, #181, #235, #241, #605) — these are model traits, choose V1.2 vs V1.4 accordingly.
- CodeFormer was never part of this repo (#266).

## :zap: Quick Inference

Download a pre-trained model (or let the script auto-download it; see
[Troubleshooting](#ambulance-troubleshooting--compatibility) for blocked networks):

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P experiments/pretrained_models
```

**Inference!**

```bash
python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.4 -s 2
```

```console
Usage: python inference_gfpgan.py -i inputs/whole_imgs -o results -v 1.4 -s 2 [options]...

  -h                   show this help
  -i input             Input image or folder. Default: inputs/whole_imgs
  -o output            Output folder. Default: results
  -v version           GFPGAN model version. Option: 1 | 1.2 | 1.3 | 1.4 | RestoreFormer. Default: 1.3
  -s upscale           The final upsampling scale of the image. Default: 2
  -w weight            Adjustable weights (no effect on the released models, kept for API compatibility)
  --bg_upsampler       background upsampler. Default: realesrgan
  --bg_tile            Tile size for background sampler, 0 for no tile during testing. Default: 400
  --suffix             Suffix of the restored faces
  --only_center_face   Only restore the center face
  --aligned            Input are aligned faces
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
  --model_path         Path to a local GFPGAN model file; overrides the -v auto-download
  --fp32               Disable half precision in the background upsampler (fixes black background on some GPUs)
  --device             Device to run on, e.g. cuda, cuda:1 or cpu. Default: cuda if available, else cpu
  --seed               Seed Python, NumPy and PyTorch RNGs for reproducible batches
  --no_parse           Skip the face parsing model (slightly faster)
```

If you want to use the original model in our paper, please see [PaperModel.md](PaperModel.md) for installation and inference.

## :european_castle: Model Zoo

| Version | Model Name  | Description |
| :---: | :---:        |     :---:      |
| V1.3 | [GFPGANv1.3.pth](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth) | Based on V1.2; **more natural** restoration results; better results on very low-quality / high-quality inputs. |
| V1.2 | [GFPGANCleanv1-NoCE-C2.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth) | No colorization; no CUDA extensions are required. Trained with more data with pre-processing. |
| V1 | [GFPGANv1.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth) | The paper model, with colorization. |

The comparisons are in [Comparisons.md](Comparisons.md).

Note that V1.3 is not always better than V1.2. You may need to select different models based on your purpose and inputs.

| Version | Strengths  | Weaknesses |
| :---: | :---:        |     :---:      |
|V1.3 |  ✓ natural outputs<br> ✓better results on very low-quality inputs <br> ✓ work on relatively high-quality inputs <br>✓ can have repeated (twice) restorations | ✗ not very sharp <br> ✗ have a slight change on identity |
|V1.2 |  ✓ sharper output <br> ✓ with beauty makeup | ✗ some outputs are unnatural |

You can find **more models (such as the discriminators)** here: [[Google Drive](https://drive.google.com/drive/folders/17rLiFzcUMoQuhLnptDsKolegHWwJOnHu?usp=sharing)], OR [[Tencent Cloud 腾讯微云](https://share.weiyun.com/ShYoCCoc)]

## :computer: Training

We provide the training codes for GFPGAN (used in our paper). <br>
You could improve it according to your own needs.

**Tips**

1. More high quality faces can improve the restoration quality.
2. You may need to perform some pre-processing, such as beauty makeup.

**Procedures**

(You can try a simple version ( `options/train_gfpgan_v1_simple.yml`) that does not require face component landmarks.)

1. Dataset preparation: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

1. Download pre-trained models and other data. Put them in the `experiments/pretrained_models` folder.
    1. [Pre-trained StyleGAN2 model: StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth)
    1. [Component locations of FFHQ: FFHQ_eye_mouth_landmarks_512.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/FFHQ_eye_mouth_landmarks_512.pth)
    1. [A simple ArcFace model: arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)

1. Modify the configuration file `options/train_gfpgan_v1.yml` accordingly.

1. Training

> python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/train_gfpgan_v1.yml --launcher pytorch

## :scroll: License and Acknowledgement

GFPGAN is released under Apache License Version 2.0.

## BibTeX

    @InProceedings{wang2021gfpgan,
        author = {Xintao Wang and Yu Li and Honglun Zhang and Ying Shan},
        title = {Towards Real-World Blind Face Restoration with Generative Facial Prior},
        booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

## :e-mail: Contact

If you have any question, please email `xintao.wang@outlook.com` or `xintaowang@tencent.com`.
