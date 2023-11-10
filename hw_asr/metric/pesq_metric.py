from torchmetrics.audio import PerceptualEvaluationSpeechQuality

from hw_asr.base.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, fs: int, mode: str, n_processes: int, *args, **kwargs):
        self.pesq = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode, n_processes=n_processes)
        super().__init__(*args, **kwargs)

    def __call__(self, **batch):
        predictions = batch["short"]
        target = batch["target"]
        return self.pesq(predictions, target)
