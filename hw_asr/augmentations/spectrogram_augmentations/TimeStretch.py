from random import random

import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class myTimeStretch(AugmentationBase):
    def __init__(self, p: float, n_freq: int = 128, *args, **kwargs):
        self._p = p
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq, *args, **kwargs)  # 3 augmentation

    def __call__(self, data: Tensor):
        if random() < self._p:
            print(data.shape)
            return self._aug(data)
        return data
