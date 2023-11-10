from random import uniform

from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Masking(AugmentationBase):
    def __init__(self, p: float, freq_max: int = 20, time_max: int = 30, *args, **kwargs):
        self._p = p
        self._freq_aug = FrequencyMasking(freq_max, *args, **kwargs)  # 1 augmentation
        self._time_aug = TimeMasking(time_max, *args, **kwargs)  # 2 augmentation

    def __call__(self, data: Tensor):
        if uniform(0, 1) < self._p:
            return self._time_aug(self._freq_aug(data))
        return data
