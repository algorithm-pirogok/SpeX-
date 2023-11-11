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

    def _compute_sisdr(self, estimate, target):
        if estimate.shape != target.shape:
            raise ValueError("Inputs must have the same shape")

        # Computing the numerator and denominator components of the SI-SDR
        product = torch.sum(target * estimate, dim=-1, keepdim=True) * target / torch.sum(target ** 2, dim=-1,
                                                                                          keepdim=True)
        error = estimate - product

        # Computing SI-SDR
        SISDR = 10 * torch.log10(torch.sum(product ** 2, dim=-1) / torch.sum(error ** 2, dim=-1))

        # Averaging SI-SDR over the batch
        loss = -torch.mean(SISDR)
        print("PRED_LOSS:", loss)
        print("LOSS:", self.si_sdr)
        print(estimate)
        product("REAL LOSS:", self.si_sdr(estimate.to("cpu").detach(), target.to("cpu").detach()))
        return loss

    def forward(self, short_pred, middle_pred, long_pred, target):
        short = self._compute_sisdr(short_pred, target)
        middle = self._compute_sisdr(middle_pred, target)
        long = self._compute_sisdr(long_pred, target)
        return self.short_coeff * short + self.middle_coeff * middle + self.long_coeff * long


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
