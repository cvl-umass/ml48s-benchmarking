import torch
import torch.nn as nn

# https://arxiv.org/pdf/2203.16219

class EntropyMaximizationLoss(nn.Module):
    """Use for target-only! Assumes all negative samples are unnanotated"""
    def __init__(self, alpha=0.1):
        super(EntropyMaximizationLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x, y, mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        orig_loss = nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        fc = torch.sigmoid(x[mask == 0])
        orig_loss[mask == 0] = self.alpha * (fc * torch.log(fc + 1e-5) + (1-fc) * torch.log(1-fc + 1e-5))
        return orig_loss
