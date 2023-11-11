import torch.nn as nn
import torch


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

    def _compute_sisdr(self, est_batch, target_batch):
        alpha = (target_batch * est_batch).sum(dim=-1) / torch.norm(target_batch, dim=1) ** 2
        scale_target = alpha[:, None] * target_batch
        return 20 * torch.log10(
            torch.sum(torch.norm(scale_target, dim=1) / (torch.norm(scale_target - est_batch, dim=1) + self.eps)
                      + self.eps))

    def forward(self, short_pred, middle_pred, long_pred, target):
        short = self._compute_sisdr(short_pred, target)
        middle = self._compute_sisdr(middle_pred, target)
        long = self._compute_sisdr(long_pred, target)
        return -(self.short_coeff * short + self.middle_coeff * middle + self.long_coeff * long) / short_pred.shape[0]


class FinalLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float, eps: float):
        super(FinalLoss, self).__init__()
        self.sisdr = SI_SDR(alpha, beta, eps)

        self.gamma = gamma
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, short_pred, middle_pred, long_pred, target, log_probs, speaker_id):
        print("START_COMPUTE_LOSS")
        sisdr = self.sisdr(short_pred, middle_pred, long_pred, target).to("cuda")
        ce_loss = self.celoss(log_probs, torch.Tensor(speaker_id).type(torch.LongTensor).to("cuda")).detach()
        loss = sisdr + self.gamma * ce_loss
        print("END_COMPUTE_LOSS")
        return loss


