import argparse
import collections
import warnings

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch

import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
import hw_asr.model as module_arch
from hw_asr.trainer import Trainer
from hw_asr.utils import prepare_device
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser

# Отключение предупреждений
warnings.simplefilter("ignore", UserWarning)

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path='hw_asr/configs', config_name='config')
def main(clf: DictConfig):
    config = ConfigParser(clf)
    logger = config.get_logger("train")
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)
    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric, text_encoder=text_encoder)
        for metric_dict in config["metrics"]
    ]
    metrics_test = metrics
    metrics_train = [
        metric for metric in metrics if metric.train
    ]
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(
        model,
        loss_module,
        metrics_train,
        metrics_test,
        optimizer,
        text_encoder=text_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )
    trainer.train()


if __name__ == "__main__":
    main()
