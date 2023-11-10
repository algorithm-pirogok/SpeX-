from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_asr.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean: bool, *args, **kwargs):
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        predictions = batch["short"].to("cpu").detach()
        target = batch["target"].to("cpu").detach()
        return self.si_sdr(predictions, target)
