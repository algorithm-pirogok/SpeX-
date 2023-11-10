import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    for name in ('snr', 'speaker_id', 'audio_len'):
        result_batch[name] = [item[name] for item in dataset_items]

    for name in ['mixed', 'ref', 'target']:
        result_batch[name] = torch.stack([item[name][0] for item in dataset_items])

    return result_batch
