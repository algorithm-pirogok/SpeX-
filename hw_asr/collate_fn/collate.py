import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {'audio_len': torch.tensor([max(item['mixed'].shape[1], item['ref'].shape[1])
                                               for item in dataset_items]),
                    'ref_len': torch.tensor([item['ref'].shape[1] for item in dataset_items]),
                    'speaker_id': [item['speaker_id'] for item in dataset_items],
                    'snr': [item['snr'] for item in dataset_items]}

    items_len = len(dataset_items)
    audio_len = max([max(item['mixed'].shape[1], item['ref'].shape[1]) for item in dataset_items])
    ref_len = max([item['ref'].shape[1] for item in dataset_items])
    for name, length in zip(['mixed', 'ref', 'target'], [audio_len, ref_len, audio_len]):
        result_batch[name] = torch.zeros(items_len, length)
        for index, item in enumerate(dataset_items):
            curr_len = item[name].shape[-1]
            result_batch[name][index, :curr_len] = item[name].squeeze()

    return result_batch
