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


ensure_functional_tensor_shim()
