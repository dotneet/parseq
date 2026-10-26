# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import numpy as np
from PIL import Image, ImageFilter

from timm.data import auto_augment

from strhub.data import aa_overrides

aa_overrides.apply()

_OP_CACHE = {}


def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))


def gaussian_blur(img, radius, **__):
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)


def _motion_blur_kernel(k: int, angle: float, direction: float) -> np.ndarray:
    """Line kernel of size k x k, rotated by `angle` degrees.

    `direction` in [-1, 1] shifts the weight towards one end of the line (0 = uniform).
    Mirrors the kernel used by imgaug's MotionBlur.
    """
    weights = np.linspace(1.0 - direction, 1.0 + direction, k).clip(min=0)
    weights /= weights.sum()
    line = np.zeros((k, k), dtype=np.float32)
    line[k // 2, :] = weights
    kernel = Image.fromarray(line, mode='F').rotate(angle, resample=Image.Resampling.BILINEAR)
    kernel = np.asarray(kernel, dtype=np.float32)
    total = kernel.sum()
    return kernel / total if total > 0 else line


def _convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0]
    pad = k // 2
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(0, 1))
    out = np.einsum('hwcij,ij->hwc', windows, kernel)
    return out.reshape(img.shape)


def _add_noise(img: Image.Image, noise: np.ndarray) -> Image.Image:
    arr = np.asarray(img)
    if arr.ndim == 3:
        noise = noise[..., None]  # Same noise for all channels (per_channel=False)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def motion_blur(img, k, **__):
    k = _get_param(k, img, 0.08, 3) | 1  # bin to odd values
    kernel = _motion_blur_kernel(k, np.random.uniform(0, 360), np.random.uniform(-1.0, 1.0))
    arr = _convolve(np.asarray(img), kernel)
    return Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8))


def gaussian_noise(img, scale, **_):
    scale = _get_param(scale, img, 0.25) | 1  # bin to odd values
    noise = np.random.normal(0.0, scale, size=(img.height, img.width))
    return _add_noise(img, np.round(noise))


def poisson_noise(img, lam, **_):
    lam = _get_param(lam, img, 0.2) | 1  # bin to odd values
    noise = np.random.poisson(lam, size=(img.height, img.width))
    sign = np.random.choice([-1, 1], size=noise.shape)
    return _add_noise(img, sign * noise)


def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return (level,)


_RAND_TRANSFORMS = auto_augment._RAND_INCREASING_TRANSFORMS.copy()
_RAND_TRANSFORMS.remove('SharpnessIncreasing')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    # 'MotionBlur',
    # 'GaussianNoise',
    'PoissonNoise',
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur': partial(_level_to_arg, max=4),
    'MotionBlur': partial(_level_to_arg, max=20),
    'GaussianNoise': partial(_level_to_arg, max=0.1 * 255),
    'PoissonNoise': partial(_level_to_arg, max=40),
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise,
})


def rand_augment_transform(magnitude=5, num_layers=3):
    # These are tuned for magnitude=5, which means that effective magnitudes are half of these values.
    hparams = {
        'rotate_deg': 30,
        'shear_x_pct': 0.9,
        'shear_y_pct': 0.2,
        'translate_x_pct': 0.10,
        'translate_y_pct': 0.30,
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude, hparams=hparams, transforms=_RAND_TRANSFORMS)
    # Supply weights to disable replacement in random selection (i.e. avoid applying the same op twice)
    choice_weights = [1.0 / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)
