import torch
import torch.nn as nn

# https://arxiv.org/pdf/2203.16219

class EntropyMaximizationLoss(nn.Module):
    """Use for target-only! Assumes all negative samples are unnanotated"""
    def __init__(self, alpha=0.1, beta=0, balance=1):
        super(EntropyMaximizationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.balance = balance

    def forward(self, x, y, mask, pseudo=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        if self.balance == 999:
            mask = mask.clone()
            mask[(mask == 1) & (y == 0)] = 0
        orig_loss = nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        fc = torch.sigmoid(x[mask == 0])
        orig_loss[mask == 0] = self.alpha * (fc * torch.log(fc + 1e-5) + (1-fc) * torch.log(1-fc + 1e-5))
        if self.balance is None:
            orig_loss[(y == 0) & (mask == 1)] = orig_loss[(y == 0) & (mask == 1)] * (((y == 1) & (mask == 1)).sum() / ((y == 0) & (mask == 1)).sum())
        else:
            orig_loss[(y == 0) & (mask == 1)] = orig_loss[(y == 0) & (mask == 1)] * self.balance
        if self.beta > 0 and pseudo is not None:
            fc2 = torch.sigmoid(x[pseudo != 0])
            sc = pseudo[pseudo != 0]
            orig_loss[pseudo != 0] = self.beta * (sc * torch.log(fc2 + 1e-5) + (1-sc) * torch.log(1-fc2 + 1e-5))
        return torch.mean(orig_loss)
