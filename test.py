import argparse
from collections import defaultdict
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
import hydra
from hydra.utils import instantiate
import torch
from tqdm import tqdm
import pyloudnorm as pyln

from hw_asr.metric.utils import calc_cer, calc_wer
import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.metric import PESQMetric, SISDRMetric
from hw_asr.utils import ROOT_PATH, get_logger
from hw_asr.utils.object_loading import get_dataloaders

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


@hydra.main(config_path='hw_asr/configs', config_name='test_config')
def main(clf):

    logger = get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(clf)

    # build model architecture
    model = instantiate(clf["arch"])
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(clf.checkpoint))
    checkpoint = torch.load(clf.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"]
    if clf["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []
    metrcics = defaultdict(list)

    pesq = PESQMetric(name="PESQ Metric", train=False, fs=16000, mode='wb', n_processes=1, comb=False)
    comb_pesq = PESQMetric(name="Comb PESQ Metric", train=False, fs=16000, mode='wb', n_processes=1, comb=True)
    si_sdr = SISDRMetric(name="SI-SDR Metric", train=True, zero_mean=False, comb=False)
    comb_si_sdr = SISDRMetric(name="SI-SDR Metric", train=True, zero_mean=False, comb=True)

    norm_wav = pyln.Meter(clf["preprocessing"]["sr"])

    with torch.no_grad():
        for dataset in dataloaders.keys():
            for batch_num, batch in enumerate(tqdm(dataloaders[dataset])):
                batch = Trainer.move_batch_to_device(batch, device)
                output = model(**batch)
                if type(output) is dict:
                    batch.update(output)
                else:
                    raise Exception("change type of model")
                for ind in range(len(batch['snr'])):
                    for mode in ('short', 'middle', 'long'):
                        batch[mode][ind] = torch.tensor(pyln.normalize.loudness(
                            batch[mode][ind].cpu().numpy(),
                            norm_wav.integrated_loudness(batch[mode][ind].cpu().numpy()),
                            -23.0
                        )).to(device)
                    curr_batch = {key: batch[key][ind] for key in batch.keys()}
                    metrcics["si_sdr"].append(si_sdr(**curr_batch).item())
                    metrcics["comb_si_sdr"].append(comb_si_sdr(**curr_batch).item())
                    metrcics["pesq"].append(pesq(**curr_batch).item())
                    metrcics["comb_pesq"].append(comb_pesq(**curr_batch).item())

                print("Iteration:", batch_num)
                for key, value in metrcics.items():
                    print(f"{key}: {np.mean(value)}")

    final_dict = {}
    for key, value in metrcics.items():
        final_dict[key] = np.mean(value)
        print(f"{key}: {np.mean(value)}")

    with open(clf.out_file, "w") as f:
        json.dump(final_dict, f, indent=2)

    '''
            logger.info(f"butch_num {batch_num}, len_of_object {len(metrcics['text_argmax'])}")

            for key, history in metrcics.items():
                wer, cer = zip(*history)
                wer = np.mean(wer)
                cer = np.mean(cer)
                logger.info(f'{mode} {key}_WER = {wer}')
                logger.info(f'{mode} {key}_CER = {cer}')

            with Path(out_file).open("w") as f:
                json.dump(results, f, indent=2)'''


if __name__ == "__main__":
    main()
