import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from gfpgan.model_registry import download_model, ensure_runtime_models, find_model_spec

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
        device (str | torch.device): Device to run on. Default: cuda if available, else cpu.
        use_parse (bool): Use the face parsing model for better blending around
            face borders. Set to False to skip it and save a little time. Default: True.
        model_rootpath (str): Directory for verified runtime model files.
        offline (bool): Refuse all model downloads when True. Default: False.
    """

    def __init__(self,
                 model_path,
                 upscale=2,
                 arch='clean',
                 channel_multiplier=2,
                 bg_upsampler=None,
                 device=None,
                 use_parse=True,
                 model_rootpath=None,
                 offline=False):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        if arch not in {'clean', 'bilinear', 'original', 'RestoreFormer'}:
            raise ValueError(f'Unsupported GFPGAN architecture: {arch}')

        # initialize model
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:  # accept both torch.device and strings like 'cuda:1' (#474, #148)
            self.device = torch.device(device) if isinstance(device, str) else device
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'bilinear':
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'original':
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'RestoreFormer':
            from gfpgan.archs.restoreformer_arch import RestoreFormer
            self.gfpgan = RestoreFormer()
        # initialize face helper
        # anchor the facexlib weight directory to this package (absolute path);
        # a relative path makes facexlib download to the current working
        # directory (#442, #389) and re-download when run from elsewhere.
        model_rootpath = os.path.abspath(model_rootpath or os.path.join(ROOT_DIR, 'gfpgan/weights'))
        # facexlib performs its own downloads. Populate the exact filenames
        # with verified official artifacts so those fallback downloads are not
        # reached. facexlib 0.3.0 initializes ParseNet even when use_parse is
        # false, so both support files remain required for compatibility.
        ensure_runtime_models(model_rootpath, use_parse=True, offline=offline)
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=use_parse,
            device=self.device,
            model_rootpath=model_rootpath)

        model_path = os.fspath(model_path)
        if model_path.startswith('https://'):
            spec = find_model_spec(model_path)
            if spec is None:
                raise ValueError(
                    'Remote GFPGAN checkpoints must be registered with a pinned SHA-256. '
                    'Download custom checkpoints yourself and pass a local path.')
            model_path = str(download_model(spec, model_rootpath, offline=offline))
        # map_location: checkpoints saved from GPU tensors must also load on
        # CPU-only machines; weights_only: the safe loader is the default since
        # torch 2.6 and the released files only contain state dicts.
        try:
            loadnet = torch.load(model_path, map_location='cpu', weights_only=True)
        except TypeError:  # torch < 1.13 has no weights_only argument
            loadnet = torch.load(model_path, map_location='cpu')
        except Exception as error:
            raise RuntimeError(f'Failed to load the GFPGAN model file "{model_path}". The download is most likely '
                               f'corrupted or incomplete (errors like "PytorchStreamReader failed reading zip archive" '
                               f'indicate this); please delete it and re-download. Original error: {error}') from error
        if not isinstance(loadnet, dict) or ('params_ema' not in loadnet and 'params' not in loadnet):
            raise RuntimeError(
                f'The file "{model_path}" is not a valid GFPGAN checkpoint (no "params"/"params_ema" found). '
                'It is either corrupted, incomplete, or not a GFPGAN model.')
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5):
        self.face_helper.clean_all()

        if weight != 0.5:
            # No released architecture consumes this option (it is swallowed by
            # **kwargs in every forward); warn instead of failing silently.
            print('Note: `weight` has no effect on the released GFPGAN models '
                  '(no architecture consumes it). The argument is kept for API compatibility.')

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
