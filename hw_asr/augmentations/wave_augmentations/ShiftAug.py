import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class ShiftAugmentation(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torch_audiomentations.PitchShift(sample_rate=16000, *args, **kwargs)  # 4 augmentation

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

