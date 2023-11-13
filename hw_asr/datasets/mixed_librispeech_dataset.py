import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import torchaudio
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.datasets.create_mixed_librispeech_dataset import load_librispeech_dataset
from hw_asr.utils import ROOT_PATH

logger = logging.getLogger(__name__)

FOLDERS = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)


class MixedLibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):

        assert part in FOLDERS or part == 'train_all', part

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "mixed_librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in FOLDERS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            load_librispeech_dataset(part, nspeakers=125, nfiles=8000)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                flac_dirs.add(dirpath)
        with open(split_dir / "speaker_to_id.json", 'r') as f:
            speaker_to_id = json.load(f)
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing mixed-librispeech folders: {part}"
        ):
            snr = float(flac_dir.split('/')[-2])
            flac_dir = Path(flac_dir)
            mixed_info = torchaudio.info(str(flac_dir / "mixed.wav"))
            ref_info = torchaudio.info(str(flac_dir / "ref.wav"))
            index.append(
                {
                    "snr": snr,
                    "speaker_id": speaker_to_id[str(flac_dir).split('/')[-1].split('_')[0]],
                    "audio_len": mixed_info.num_frames / mixed_info.sample_rate,
                    "ref_len": ref_info.num_frames / ref_info.sample_rate,
                    "mixed": str(flac_dir / "mixed.wav"),
                    "ref": str(flac_dir / "ref.wav"),
                    "target": str(flac_dir / "target.wav"),
                    "text": "",
                    "path": ""
                }
            )
        return index


if __name__ == "__main__":
    tmp = MixedLibrispeechDataset("test-clean")
