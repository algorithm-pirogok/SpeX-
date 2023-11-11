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

    def _print_loss(self, est_batch, target_batch):
        alpha = torch.sum(target_batch * est_batch, dim=-1, keepdim=True) \
                / (torch.norm(target_batch, dim=-1, keepdim=True) ** 2 + self.eps)
        scale_target = alpha * target_batch
        print("OLD LOSS", 20 * torch.log10(
            torch.sum(torch.norm(scale_target, dim=1) / (torch.norm(scale_target - est_batch, dim=1) + self.eps)
                      + self.eps)))

        scale_target = torch.sum(target_batch * est_batch, dim=-1, keepdim=True) * target_batch \
                       / (torch.sum(target_batch ** 2, dim=-1, keepdim=True) + self.eps)

        print("V1 loss", 10 * torch.log10(
            torch.sum(torch.sum(scale_target ** 2, dim=1)
                      / (torch.sum((scale_target - est_batch) ** 2, dim=1) + self.eps)
                      + self.eps)))

        scale_target = torch.sum(target_batch * est_batch, dim=-1, keepdim=True) * target_batch \
                       / (torch.sum(target_batch ** 2, dim=-1, keepdim=True) + self.eps)
        num = torch.sum((scale_target - est_batch) ** 2, dim=-1) + self.eps
        frac = torch.sum(scale_target ** 2, dim=-1) + self.eps
        print("V2 loss", torch.sum((-10 * torch.log10(num / frac))))

    def _compute_sisdr(self, est_batch, target_batch):
        scale_target = torch.sum(target_batch * est_batch, dim=-1, keepdim=True) * target_batch \
                       / (torch.sum(target_batch ** 2, dim=-1, keepdim=True) + self.eps)
        self._print_loss(est_batch, target_batch)
        final_loss = torch.sum(10 * torch.log10(
            torch.sum(scale_target ** 2, dim=1)
            / (torch.sum((scale_target - est_batch) ** 2, dim=1) + self.eps)
            + self.eps))
        print("FINAL LOSS", final_loss)
        return final_loss

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
        sisdr = self.sisdr(short_pred, middle_pred, long_pred, target)
        ce_loss = self.celoss(log_probs, speaker_id)
        loss = sisdr + self.gamma * ce_loss
        return loss
