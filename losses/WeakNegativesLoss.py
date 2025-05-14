import torch
import torch.nn as nn

class WeakNegativesLoss(nn.Module):
    def __init__(self, gamma=1/100):
        super(WeakNegativesLoss, self).__init__()
        self.gamma = gamma

    def forward(self, x, y, mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        orig_loss = nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        orig_loss[mask == 0] = orig_loss[mask == 0] * self.gamma
        return orig_loss
