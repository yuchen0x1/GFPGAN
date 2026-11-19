# flake8: noqa
# This file is used for deploying replicate models
# running: cog predict -i img=@inputs/whole_imgs/10045.png -i version='v1.4' -i scale=2
# push: cog push r8.im/tencentarc/gfpgan
# push (backup): cog push r8.im/xinntao/gfpgan

import cv2
import numpy as np
import os
import tempfile
import torch
from basicsr.archs.srvgg_arch import SRVGGNetCompact

from gfpgan import GFPGANer
from gfpgan.model_registry import download_model, get_model_spec

try:
    from cog import BasePredictor, Input, Path
    from realesrgan.utils import RealESRGANer
except Exception:
    print('please install cog and realesrgan package')


class Predictor(BasePredictor):

    def setup(self):
        self.model_dir = os.environ.get('GFPGAN_MODEL_DIR', 'gfpgan/weights')
        self.model_paths = {
            key: str(download_model(get_model_spec(key), self.model_dir))
            for key in ('realesr-general-x4v3', '1.2', '1.3', '1.4', 'RestoreFormer', 'retinaface-resnet50',
                        'parsenet')
        }

        # background enhancer with RealESRGAN
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        model_path = self.model_paths['realesr-general-x4v3']
        half = True if torch.cuda.is_available() else False
        self.upsampler = RealESRGANer(
            scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

        # Use GFPGAN for face enhancement
        self.face_enhancer = GFPGANer(
            model_path=self.model_paths['1.4'],
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
            model_rootpath=self.model_dir)
        self.current_version = 'v1.4'

    def predict(
            self,
            img: Path = Input(description='Input'),
            version: str = Input(
                description='GFPGAN version. v1.3: better quality. v1.4: more details and better identity.',
                choices=['v1.2', 'v1.3', 'v1.4', 'RestoreFormer'],
                default='v1.4'),
            scale: float = Input(description='Rescaling factor', default=2),
    ) -> Path:
        weight = 0.5
        print(img, version, scale, weight)
        extension = os.path.splitext(os.path.basename(str(img)))[1].lstrip('.').lower() or 'png'
        img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError('The input is not a readable image.')
        output = None
        try:
            alpha = None
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
                # the detection / restoration networks only accept 3 channels;
                # keep the alpha aside and re-attach it to the output image
                alpha = img[:, :, 3]
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif len(img.shape) == 2:
                img_mode = None
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_mode = None

            h, w = img.shape[0:2]
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            if self.current_version != version:
                if version == 'v1.2':
                    self.face_enhancer = GFPGANer(
                        model_path=self.model_paths['1.2'],
                        upscale=2,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler,
                        model_rootpath=self.model_dir)
                    self.current_version = 'v1.2'
                elif version == 'v1.3':
                    self.face_enhancer = GFPGANer(
                        model_path=self.model_paths['1.3'],
                        upscale=2,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler,
                        model_rootpath=self.model_dir)
                    self.current_version = 'v1.3'
                elif version == 'v1.4':
                    self.face_enhancer = GFPGANer(
                        model_path=self.model_paths['1.4'],
                        upscale=2,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler,
                        model_rootpath=self.model_dir)
                    self.current_version = 'v1.4'
                elif version == 'RestoreFormer':
                    self.face_enhancer = GFPGANer(
                        model_path=self.model_paths['RestoreFormer'],
                        upscale=2,
                        arch='RestoreFormer',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler,
                        model_rootpath=self.model_dir)
                    self.current_version = version

            try:
                _, _, output = self.face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True, weight=weight)
            except RuntimeError as error:
                print('Error', error)
                # keep the run alive: fall back to the (unrestored) input so an
                # output image is still produced instead of an UnboundLocalError
                output = img

            try:
                if scale != 2:
                    interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
                    h, w = img.shape[0:2]
                    output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
            except Exception as error:
                print('wrong scale input.', error)

            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
                if output.shape[:2] != alpha.shape[:2]:
                    alpha = cv2.resize(alpha, (output.shape[1], output.shape[0]), interpolation=cv2.INTER_LINEAR)
                output = np.concatenate([output, alpha[:, :, np.newaxis]], axis=2)
            # save_path = f'output/out.{extension}'
            # cv2.imwrite(save_path, output)
            out_path = Path(tempfile.mkdtemp()) / f'out.{extension}'
            if not cv2.imwrite(str(out_path), output):
                raise IOError(f'Failed to encode output image: {out_path}')
        except Exception as error:
            raise RuntimeError(f'GFPGAN prediction failed: {error}') from error
        return out_path
