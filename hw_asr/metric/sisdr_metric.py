from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_asr.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean: bool, *args, **kwargs):
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        short = batch["short"].to("cpu").detach()
        middle = batch["middle"].to("cpu").detach()
        long = batch["long"].to("cpu").detach()
        target = batch["target"].to("cpu").detach()
        pred = short * 0.8 + middle * 0.1 + long * 0.1
        return self.si_sdr(pred, target)
