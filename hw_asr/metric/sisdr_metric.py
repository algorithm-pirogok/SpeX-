from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from hw_asr.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean: bool, *args, **kwargs):
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=zero_mean)
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        short = batch["short"].to("cpu").detach()
        middle = batch["middle"].to("cpu").detach()
        target = batch["target"].to("cpu").detach()
        ref = batch["ref"].to("cpu").detach()
        print("SHORT/MIDDLE SI_SDR:", self.si_sdr(short, middle))
        print("SHORT/target SI_SDR:", self.si_sdr(short, target))
        print("SHORT/ref SI_SDR:", self.si_sdr(short, ref))
        return self.si_sdr(short, target)
