import cv2
import os
import pytest
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from gfpgan.utils import GFPGANer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_MODEL = os.path.join(REPO_ROOT, 'experiments', 'pretrained_models', 'GFPGANCleanv1-NoCE-C2.pth')
V1_MODEL = os.path.join(REPO_ROOT, 'experiments', 'pretrained_models', 'GFPGANv1.pth')
has_models = os.path.isfile(CLEAN_MODEL) and os.path.isfile(V1_MODEL)


@pytest.mark.skipif(not has_models, reason='pretrained models not downloaded into experiments/pretrained_models')
def test_gfpganer():
    # initialize with the clean model
    restorer = GFPGANer(model_path=CLEAN_MODEL, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None)
    # test attribute
    assert isinstance(restorer.gfpgan, GFPGANv1Clean)
    assert isinstance(restorer.face_helper, FaceRestoreHelper)

    # initialize with the original model
    restorer = GFPGANer(model_path=V1_MODEL, upscale=2, arch='original', channel_multiplier=1, bg_upsampler=None)
    # test attribute
    assert isinstance(restorer.gfpgan, GFPGANv1)
    assert isinstance(restorer.face_helper, FaceRestoreHelper)

    # ------------------ test enhance ---------------- #
    img = cv2.imread(os.path.join(REPO_ROOT, 'tests', 'data', 'gt', '00000000.png'), cv2.IMREAD_COLOR)
    result = restorer.enhance(img, has_aligned=False, paste_back=True)
    assert result[0][0].shape == (512, 512, 3)
    assert result[1][0].shape == (512, 512, 3)
    assert result[2].shape == (1024, 1024, 3)

    # with has_aligned=True
    result = restorer.enhance(img, has_aligned=True, paste_back=False)
    assert result[0][0].shape == (512, 512, 3)
    assert result[1][0].shape == (512, 512, 3)
    assert result[2] is None
