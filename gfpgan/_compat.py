"""Compatibility shims for GFPGAN across PyTorch / torchvision versions.

Import this module (or simply ``import gfpgan``) before importing
``basicsr`` or ``realesrgan`` so that the pinned basicsr release keeps
working with recent torchvision versions.
"""
import sys


def ensure_functional_tensor_shim():
    """Keep basicsr importable on torchvision >= 0.17.

    basicsr 1.4.2 imports ``rgb_to_grayscale`` from the private module
    ``torchvision.transforms.functional_tensor``, which was removed in
    torchvision 0.17 (upstream issues #506, #510, #539, #556, #602, #603).
    Register a small shim module that re-exports the function from its new
    location so the existing import keeps working.
    """
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ImportError:
        pass
    try:
        from torchvision.transforms import functional as _functional
    except ImportError:
        return
    import types

    shim = types.ModuleType('torchvision.transforms.functional_tensor')
    shim.rgb_to_grayscale = _functional.rgb_to_grayscale
    sys.modules.setdefault('torchvision.transforms.functional_tensor', shim)


def patch_stylegan2_ops_native_fallback():
    """Give basicsr's StyleGAN2 CUDA ops a pure-PyTorch fallback.

    basicsr 1.4.2 only defines ``fused_act_ext`` / ``upfirdn2d_ext`` when the
    compiled extension is present (or BASICSR_JIT=True), yet its forward
    paths call them unconditionally, so the GFPGANv1 paper architecture
    crashes with ``NameError: name 'fused_act_ext' is not defined`` (#608,
    #96, #115, #190) and cannot run on machines without compiled extensions
    or on CPU (#81, #384). When the extension is absent, install equivalent
    native implementations (adapted from rosinality/stylegan2-pytorch), so
    the same math runs everywhere with plain autograd. The compiled path is
    kept untouched when the extension exists.

    Importing ``basicsr`` executes ``basicsr.archs.stylegan2_arch``, which
    binds the original functions at import time, so after installing the
    fallback the consumer module's namespace is re-bound as well.
    """
    import torch.nn.functional as _F

    # NOTE: `import basicsr.ops.upfirdn2d.upfirdn2d as X` would bind the
    # re-exported *function* (the package attribute shadows the submodule),
    # so resolve the real submodule files through importlib instead.
    import importlib

    try:
        _fused_act = importlib.import_module('basicsr.ops.fused_act.fused_act')
        _fused_act_pkg = importlib.import_module('basicsr.ops.fused_act')
    except ImportError:
        _fused_act = None
        _fused_act_pkg = None
    try:
        _upfirdn2d = importlib.import_module('basicsr.ops.upfirdn2d.upfirdn2d')
        _upfirdn2d_pkg = importlib.import_module('basicsr.ops.upfirdn2d')
    except ImportError:
        _upfirdn2d = None
        _upfirdn2d_pkg = None
    if _fused_act is None or _upfirdn2d is None:
        return

    if not hasattr(_fused_act, 'fused_act_ext'):

        def _native_fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
            # Broadcast the bias exactly like the CUDA kernel does
            # (bias index = (flat_idx // prod(size[2:])) % bias.numel()):
            # per-channel on dim 1 for conv inputs (>= 4D), per-last-dim for
            # linear inputs (2D / 3D).
            if input.ndim >= 4:
                return _F.leaky_relu(
                    input + bias.view(1, -1, *([1] * (input.ndim - 2))), negative_slope=negative_slope) * scale
            return _F.leaky_relu(
                input + bias.view(*([1] * (input.ndim - 1)), -1), negative_slope=negative_slope) * scale

        _fused_act.fused_leaky_relu = _native_fused_leaky_relu
        _fused_act.FusedLeakyReLU.forward = lambda self, input: _native_fused_leaky_relu(
            input, self.bias, self.negative_slope, self.scale)
        if _fused_act_pkg is not None:
            _fused_act_pkg.fused_leaky_relu = _native_fused_leaky_relu

    if not hasattr(_upfirdn2d, 'upfirdn2d_ext'):

        def _upfirdn2d_dispatch(input, kernel, up=1, down=1, pad=(0, 0)):
            return _upfirdn2d.upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

        _upfirdn2d.upfirdn2d = _upfirdn2d_dispatch
        if _upfirdn2d_pkg is not None:
            _upfirdn2d_pkg.upfirdn2d = _upfirdn2d_dispatch

    # re-bind in the consumer module that captured the originals at import time
    try:
        import basicsr.archs.stylegan2_arch as _stylegan2_arch
    except ImportError:
        return
    if not hasattr(_fused_act, 'fused_act_ext'):
        _stylegan2_arch.fused_leaky_relu = _fused_act.fused_leaky_relu
    if not hasattr(_upfirdn2d, 'upfirdn2d_ext'):
        _stylegan2_arch.upfirdn2d = _upfirdn2d.upfirdn2d


def patch_basicsr_jpeg_quality():
    """basicsr passes a float JPEG quality to cv2.imencode; OpenCV >= 4.8
    rejects non-int imencode params, which broke training with
    ``cv2.error: (-5:Bad argument) in function 'imencode'`` (#93, #222).
    Wrap ``add_jpg_compression`` so the quality is cast to int first.
    """
    try:
        from basicsr.data import degradations
    except ImportError:
        return
    orig = getattr(degradations, 'add_jpg_compression', None)
    if orig is None or getattr(orig, '_gfpgan_int_quality_patch', False):
        return
    import functools

    @functools.wraps(orig)
    def add_jpg_compression(img, quality=90):
        return orig(img, int(quality))

    add_jpg_compression._gfpgan_int_quality_patch = True
    degradations.add_jpg_compression = add_jpg_compression


ensure_functional_tensor_shim()
patch_stylegan2_ops_native_fallback()
patch_basicsr_jpeg_quality()
