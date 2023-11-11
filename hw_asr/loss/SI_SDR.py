import torch.nn as nn
import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SI_SDR(nn.Module):
    def __init__(self, short: float, middle: float, eps: float):
        """
        alpha: for short
        beta
        """
        super(SI_SDR, self).__init__()
        self.short_coeff = short
        self.middle_coeff = middle
        self.long_coeff = 1 - short - middle
        self.eps = eps
        self.si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=False)

    def _compute_sisdr(self, preds, target):
        alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + self.eps) / (
                torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        )
        target_scaled = alpha * target

        noise = target_scaled - preds

        val = (torch.sum(target_scaled ** 2, dim=-1) + self.eps) / (torch.sum(noise ** 2, dim=-1) + self.eps)
        return -10 * torch.log10(val)

    def forward(self, short_pred, middle_pred, long_pred, target):
        short = self._compute_sisdr(short_pred, target)
        middle = self._compute_sisdr(middle_pred, target)
        long = self._compute_sisdr(long_pred, target)
        return (self.short_coeff * short + self.middle_coeff * middle + self.long_coeff * long) / short_pred.shape[0]


class FinalLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float, eps: float):
        super(FinalLoss, self).__init__()
        self.sisdr = SI_SDR(alpha, beta, eps)

        self.gamma = gamma
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, short_pred, middle_pred, long_pred, target, log_probs, speaker_id):
        sisdr = self.sisdr(short_pred, middle_pred, long_pred, target)
        ce_loss = self.celoss(log_probs, speaker_id)
        loss = sisdr + self.gamma * ce_loss
        return loss
