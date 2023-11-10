import glob
from glob import glob
import random
import os

from concurrent.futures import ProcessPoolExecutor
import json
import librosa
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import warnings

from hw_asr.datasets.librispeech_dataset import LibrispeechDataset
from hw_asr.utils import ROOT_PATH

warnings.filterwarnings("ignore")

NAMES = {
    "dev-clean": True,
    "dev-other": True,
    "test-clean": True,
    "test-other": True,
    "train-clean-100": False,
    "train-clean-360": False,
    "train-other-500": False
}


class LibriSpeechSpeakerFiles:
    def __init__(self, speaker_id, audios_dir, audioTemplate="*-norm.wav"):
        self.id = speaker_id
        self.files = []
        self.audioTemplate = audioTemplate
        self.files = self.find_files_by_worker(audios_dir)

    def find_files_by_worker(self, audios_dir):
        speakerDir = os.path.join(audios_dir, self.id)  # it is a string
        chapterDirs = os.scandir(speakerDir)
        files = []
        for chapterDir in chapterDirs:
            files = files + [file for file in
                             glob(os.path.join(speakerDir, chapterDir.name) + "/" + self.audioTemplate)]
        return files


def snr_mixer(clean, noise, snr):
    amp_noise = np.linalg.norm(clean) / 10 ** (snr / 20)

    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise

    mix = clean + noise_norm

    return mix


def vad_merge(w, top_db):
    intervals = librosa.effects.split(w, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def cut_audios(s1, s2, sec, sr):
    cut_len = sr * sec
    len1 = len(s1)
    len2 = len(s2)

    s1_cut = []
    s2_cut = []

    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])

        segment += 1

    return s1_cut, s2_cut


def fix_length(s1, s2, min_or_max='max'):
    # Fix length
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:  # max
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2


def create_mix(idx, triplet, snr_levels, out_dir, test=False, sr=16000, **kwargs):
    trim_db, vad_db = kwargs["trim_db"], kwargs["vad_db"]
    audioLen = kwargs["audioLen"]

    s1_path = triplet["target"]
    s2_path = triplet["noise"]
    ref_path = triplet["reference"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    ref, _ = sf.read(os.path.join('', ref_path))

    meter = pyln.Meter(sr)  # create BS.1770 meter

    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    loudsRef = meter.integrated_loudness(ref)

    s1Norm = pyln.normalize.loudness(s1, louds1, -29)
    s2Norm = pyln.normalize.loudness(s2, louds2, -29)
    refNorm = pyln.normalize.loudness(ref, loudsRef, -23.0)

    amp_s1 = np.max(np.abs(s1Norm))
    amp_s2 = np.max(np.abs(s2Norm))
    amp_ref = np.max(np.abs(refNorm))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0:
        return

    if trim_db:
        ref, _ = librosa.effects.trim(refNorm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1Norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2Norm, top_db=trim_db)

    if len(ref) < sr:
        return

    for snr in snr_levels:
        str_snr = "{0:0.1f}".format(snr)
        path_mix = os.path.join(out_dir, f"{str_snr}/{target_id}_{noise_id}_" + "%06d" % idx + "/mixed.wav")
        path_target = os.path.join(out_dir, f"{str_snr}/{target_id}_{noise_id}_" + "%06d" % idx + "/target.wav")
        path_ref = os.path.join(out_dir, f"{str_snr}/{target_id}_{noise_id}_" + "%06d" % idx + "/ref.wav")

        if not test:
            s1, s2, ref = vad_merge(s1, vad_db), vad_merge(s2, vad_db), vad_merge(ref, vad_db)
            s1_cut, s2_cut = cut_audios(s1, s2, audioLen, sr)
            ref_cut, _ = cut_audios(ref, ref, audioLen, sr)
            if not len(ref_cut):
                return
            ref_cut = ref_cut[0]

            for i in range(len(s1_cut)):
                mix = snr_mixer(s1_cut[i], s2_cut[i], snr)

                louds1 = meter.integrated_loudness(s1_cut[i])
                s1_cut[i] = pyln.normalize.loudness(s1_cut[i], louds1, -23.0)
                loudMix = meter.integrated_loudness(mix)
                mix = pyln.normalize.loudness(mix, loudMix, -23.0)

                path_mix_i = path_mix.replace("/mixed.wav", f"_{i}/mixed.wav")
                path_target_i = path_target.replace("/target.wav", f"_{i}/target.wav")
                path_ref_i = path_ref.replace("/ref.wav", f"_{i}/ref.wav")

                directory = os.path.dirname(path_mix_i)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                sf.write(path_mix_i, mix, sr)
                sf.write(path_target_i, s1_cut[i], sr)
                sf.write(path_ref_i, ref_cut, sr)
        else:
            s1, s2 = fix_length(s1, s2, 'max')
            mix = snr_mixer(s1, s2, snr)
            louds1 = meter.integrated_loudness(s1)
            s1 = pyln.normalize.loudness(s1, louds1, -23.0)

            loudMix = meter.integrated_loudness(mix)
            mix = pyln.normalize.loudness(mix, loudMix, -23.0)

            directory = os.path.dirname(path_mix)
            if not os.path.exists(directory):
                os.makedirs(directory)

            sf.write(path_mix, mix, sr)
            sf.write(path_target, s1, sr)
            sf.write(path_ref, ref, sr)


class MixtureGenerator:
    def __init__(self, speakers_files, out_folder, nfiles=5000, test=False, randomState=42):
        self.speakers_files = speakers_files  # list of SpeakerFiles for every speaker_id
        self.nfiles = nfiles
        self.randomState = randomState
        self.out_folder = out_folder
        self.test = test
        random.seed(self.randomState)
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def generate_triplets(self):
        i = 0
        all_triplets = {"reference": [], "target": [], "noise": [], "target_id": [], "noise_id": []}
        while i < self.nfiles:
            spk1, spk2 = random.sample(self.speakers_files, 2)

            if len(spk1.files) < 2 or len(spk2.files) < 2:
                continue

            target, reference = random.sample(spk1.files, 2)
            noise = random.choice(spk2.files)
            all_triplets["reference"].append(reference)
            all_triplets["target"].append(target)
            all_triplets["noise"].append(noise)
            all_triplets["target_id"].append(spk1.id)
            all_triplets["noise_id"].append(spk2.id)
            i += 1

        return all_triplets

    def triplet_generator(self, target_speaker, noise_speaker, number_of_triplets):
        max_num_triplets = min(len(target_speaker.files), len(noise_speaker.files))
        number_of_triplets = min(max_num_triplets, number_of_triplets)

        target_samples = random.sample(target_speaker.files, k=number_of_triplets)
        reference_samples = random.sample(target_speaker.files, k=number_of_triplets)
        noise_samples = random.sample(noise_speaker.files, k=number_of_triplets)

        triplets = {"reference": [], "target": [], "noise": [],
                    "target_id": [target_speaker.id] * number_of_triplets,
                    "noise_id": [noise_speaker.id] * number_of_triplets}
        triplets["target"] += target_samples
        triplets["reference"] += reference_samples
        triplets["noise"] += noise_samples

        return triplets

    def generate_mixes(self, snr_levels=[0], num_workers=10, update_steps=10, **kwargs):

        triplets = self.generate_triplets()

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = []

            for i in range(self.nfiles):
                triplet = {"reference": triplets["reference"][i],
                           "target": triplets["target"][i],
                           "noise": triplets["noise"][i],
                           "target_id": triplets["target_id"][i],
                           "noise_id": triplets["noise_id"][i]}

                futures.append(pool.submit(create_mix, i, triplet,
                                           snr_levels, self.out_folder,
                                           test=self.test, **kwargs))

            for i, future in enumerate(futures):
                future.result()
                if (i + 1) % max(self.nfiles // update_steps, 1) == 0:
                    print(f"Files Processed | {i + 1} out of {self.nfiles}")


def create_dataset(dataset_name: str, nspeakers=100, nfiles: int = 100, test: bool = False,
                   num_workers: int = 2, snr_levels: list[int] = None):
    if snr_levels is None:
        snr_levels = [-3, 0, 3]
    path = ROOT_PATH / f"data/datasets/librispeech/{dataset_name}"
    path_mixtures = ROOT_PATH / f"data/datasets/mixed_librispeech/{dataset_name}"

    speakers = [el.name for el in os.scandir(path)][:nspeakers]
    speakers_files = [LibriSpeechSpeakerFiles(i, path, audioTemplate="*.flac") for i in speakers]

    mixer_train = MixtureGenerator(speakers_files,
                                   path_mixtures,
                                   nfiles=nfiles,
                                   test=test)

    mixer_train.generate_mixes(snr_levels=snr_levels,
                               num_workers=num_workers,
                               update_steps=100,
                               trim_db=20,
                               vad_db=20,
                               audioLen=3)

    speaker_to_id = {speaker: num for num, speaker in enumerate(speakers)}
    with open(path_mixtures / "speaker_to_id.json", 'w') as f:
        json.dump(speaker_to_id, f)


def load_librispeech_dataset(dataset_name: str, nspeakers: int = 150, nfiles: int = 12500, update: bool = False):
    assert dataset_name in NAMES, "Incorrect dataset naming"
    if dataset_name == "train_all":
        for name in NAMES:
            load_librispeech_dataset(name, nspeakers=nspeakers, nfiles=nfiles, update=False)
    else:
        if not os.path.exists(ROOT_PATH / f"data/datasets/librispeech/{dataset_name}"):
            base_loader = LibrispeechDataset(part=dataset_name)
            base_loader._load_part(part=dataset_name)
        if update or not os.path.exists(ROOT_PATH / f"data/datasets/merge_librispeech/{dataset_name}"):
            is_test = 'test' in dataset_name
            snr_levels = [-3, 0, 3] if not is_test else [0]
            create_dataset(dataset_name, nspeakers=nspeakers, nfiles=nfiles, test=is_test, num_workers=10,
                           snr_levels=snr_levels)


if __name__ == "__main__":
    create_dataset("test-clean", nspeakers=100, nfiles=1000, test=True)
