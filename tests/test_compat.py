import os
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V14_WEIGHTS = os.path.join(REPO_ROOT, 'experiments', 'pretrained_models', 'GFPGANv1.4.pth')
has_weights = os.path.isfile(V14_WEIGHTS)


def test_functional_tensor_shim():
    """The shim keeps basicsr importable on torchvision >= 0.17 (#506, #602, ...)."""
    import basicsr  # noqa: F401  must not raise
    import torchvision
    import torchvision.transforms.functional_tensor as ft

    import gfpgan  # noqa: F401  must be imported first, see README Troubleshooting

    assert ft.rgb_to_grayscale is not None
    assert torchvision.__version__


def test_stylegan2_ops_native_fallback_installed():
    """The ops fallback must be active whenever the compiled extension is absent (#608)."""
    import importlib

    fused = importlib.import_module('basicsr.ops.fused_act.fused_act')
    upfirdn = importlib.import_module('basicsr.ops.upfirdn2d.upfirdn2d')
    if hasattr(fused, 'fused_act_ext') and hasattr(upfirdn, 'upfirdn2d_ext'):
        pytest.skip('compiled basicsr extensions are present')
    from gfpgan._compat import patch_stylegan2_ops_native_fallback  # noqa: F401

    # FusedLeakyReLU must run without touching fused_act_ext
    m = fused.FusedLeakyReLU(8)
    out = m(torch.randn(2, 8, 4, 4))
    assert out.shape == (2, 8, 4, 4)


def test_native_fused_leaky_relu_broadcast_matches_kernel():
    """Native fused_leaky_relu must mirror the CUDA kernel's bias broadcast."""
    from gfpgan._compat import patch_stylegan2_ops_native_fallback

    patch_stylegan2_ops_native_fallback()
    from basicsr.ops.fused_act import fused_leaky_relu

    scale = 2**0.5
    slope = 0.2
    bias = torch.randn(7)

    # 2D linear case: bias per last dim
    x = torch.randn(3, 7)
    expected = F.leaky_relu(x + bias.view(1, -1), negative_slope=slope) * scale
    assert torch.allclose(fused_leaky_relu(x, bias, slope, scale), expected)

    # 3D case (EqualLinear applied to a (N, num_latent, C) style latent):
    # F.linear semantics — bias on the last dim
    x = torch.randn(2, 8, 7)
    expected = F.leaky_relu(x + bias.view(1, 1, -1), negative_slope=slope) * scale
    assert torch.allclose(fused_leaky_relu(x, bias, slope, scale), expected)

    # 4D conv case: bias per channel
    x = torch.randn(2, 7, 5, 5)
    expected = F.leaky_relu(x + bias.view(1, -1, 1, 1), negative_slope=slope) * scale
    assert torch.allclose(fused_leaky_relu(x, bias, slope, scale), expected)


def test_upfirdn2d_native_matches_naive_reference():
    """The routed upfirdn2d must agree with a naive loop reference."""
    from gfpgan._compat import patch_stylegan2_ops_native_fallback

    patch_stylegan2_ops_native_fallback()
    from basicsr.ops.upfirdn2d import upfirdn2d

    torch.manual_seed(0)
    x = torch.randn(1, 1, 6, 7)
    # the upfirdn2d contract expects a 2D kernel
    kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.]]) / 16.

    def naive(x, k, up, down, pad):
        h, w = x.shape[-2:]
        kh, kw = k.shape
        big = torch.zeros(h * up, w * up)
        big[::up, ::up] = x[0, 0]
        big = F.pad(big[None, None], (pad[0], pad[1], pad[0], pad[1]))[0, 0]
        oh = (h * up + pad[0] + pad[1] - kh) // down + 1
        ow = (w * up + pad[0] + pad[1] - kw) // down + 1
        out = torch.zeros(oh, ow)
        k_flip = torch.flip(k, [0, 1])  # the op applies the flipped kernel (true convolution)
        for i in range(oh):
            for j in range(ow):
                win = big[i * down:i * down + kh, j * down:j * down + kw]
                if win.shape == k.shape:
                    out[i, j] = (win * k_flip).sum()
        return out

    for up, down, pad in [(1, 1, (2, 2)), (2, 1, (2, 1)), (1, 2, (1, 1))]:
        got = upfirdn2d(x, kernel, up=up, down=down, pad=pad)[0, 0]
        want = naive(x, kernel, up, down, pad)
        assert torch.allclose(got, want, atol=1e-5), f'mismatch for up={up} down={down} pad={pad}'


def test_gfpganv1_paper_arch_runs_on_cpu():
    """The paper architecture must run without compiled CUDA extensions (#81, #384)."""
    from gfpgan.archs.gfpganv1_arch import GFPGANv1

    net = GFPGANv1(
        out_size=32,
        num_style_feat=64,
        channel_multiplier=1,
        resample_kernel=(1, 3, 3, 1),
        decoder_load_path=None,
        fix_decoder=True,
        num_mlp=8,
        input_is_latent=False,
        different_w=True,
        narrow=1,
        sft_half=True).cpu().eval()
    with torch.no_grad():
        output = net(torch.rand((1, 3, 32, 32)))
    assert output[0].shape == (1, 3, 32, 32)


def test_unicode_safe_image_io(tmp_path):
    """Non-ASCII paths must work on all platforms (#438)."""
    import sys

    sys.path.insert(0, REPO_ROOT)
    from inference_gfpgan import imread, imwrite

    img = torch.randint(0, 255, (8, 8, 3), dtype=torch.uint8).numpy()
    path = tmp_path / '测试图片_éè.png'
    imwrite(img, str(path))
    assert path.exists()
    back = imread(str(path))
    assert back is not None and back.shape == (8, 8, 3)


def test_cli_seed_is_parsed():
    from gfpgan import cli

    args = cli.build_parser().parse_args(['--list-models', '--seed', '123'])
    assert args.seed == 123


@pytest.mark.skipif(not has_weights, reason='GFPGANv1.4.pth not downloaded')
def test_gfpganer_device_and_corrupt_model():
    """Device strings are accepted and corrupted models raise clear errors."""
    from gfpgan import GFPGANer

    restorer = GFPGANer(model_path=V14_WEIGHTS, upscale=2, arch='clean', channel_multiplier=2, device='cpu')
    assert str(restorer.device) == 'cpu'

    corrupt = os.path.join(REPO_ROOT, 'results', '_corrupt_model.pth')
    os.makedirs(os.path.dirname(corrupt), exist_ok=True)
    with open(corrupt, 'wb') as f:
        f.write(b'definitely not a state dict')
    try:
        with pytest.raises(RuntimeError, match='corrupted|invalid'):
            GFPGANer(model_path=corrupt, upscale=2, arch='clean', channel_multiplier=2, device='cpu')
    finally:
        os.remove(corrupt)


@pytest.mark.skipif(not has_weights, reason='GFPGANv1.4.pth not downloaded')
def test_gfpganer_use_parse_false():
    """use_parse=False must still produce a restored face."""
    import cv2
    import numpy as np

    from gfpgan import GFPGANer

    img_path = os.path.join(REPO_ROOT, 'inputs', 'whole_imgs', '00.jpg')
    if not os.path.isfile(img_path):
        pytest.skip('sample image not available')
    restorer = GFPGANer(
        model_path=V14_WEIGHTS, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, use_parse=False)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    _, restored_faces, _ = restorer.enhance(img, has_aligned=False, only_center_face=True, paste_back=False)
    assert len(restored_faces) == 1
    assert restored_faces[0].shape[:2] == (512, 512)
    assert isinstance(restored_faces[0], np.ndarray)
