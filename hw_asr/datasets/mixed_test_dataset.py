import json
import logging
import os
from pathlib import Path

import torchaudio
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH

logger = logging.getLogger(__name__)


class MixedTestDataset(BaseDataset):
    def __init__(self, path_to_mix_dir: str, path_to_ref_dir: str, path_to_target_dir: str, absolute_path: bool,
                 *args, **kwargs):

        if absolute_path:
            self.path_to_mix = ROOT_PATH / path_to_mix_dir
            self.path_to_ref = ROOT_PATH / path_to_ref_dir
            self.path_to_target = ROOT_PATH / path_to_target_dir
        else:
            self.path_to_mix = Path(path_to_mix_dir)
            self.path_to_ref = Path(path_to_ref_dir)
            self.path_to_target = Path(path_to_target_dir)

        self._data_dir = ROOT_PATH / "data" / "datasets" / "test_dataset"

        index = self._get_or_load_index()

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        index_path = self._data_dir / f"index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []

        flac_files = set()
        for dirpath, dirnames, filenames in os.walk(str(self.path_to_mix)):
            for f in filenames:
                if f.endswith(".wav"):
                    flac_files.add(str(Path(dirpath) / f))

        for flac_file in tqdm(
                list(flac_files), desc=f"Preparing mixed-librispeech custom folders"
        ):
            snr = 0
            mixed_info = torchaudio.info(str(self.path_to_mix / flac_file))
            ref_info = torchaudio.info(str(self.path_to_ref / flac_file))
            index.append(
                {
                    "snr": snr,
                    "speaker_id": 0,
                    "audio_len": mixed_info.num_frames / mixed_info.sample_rate,
                    "ref_len": ref_info.num_frames / ref_info.sample_rate,
                    "mixed": str(self.path_to_mix / flac_file),
                    "ref": str(self.path_to_ref / flac_file),
                    "target": str(self.path_to_target / flac_file),
                    "text": "",
                    "path": ""
                }
            )
        return index