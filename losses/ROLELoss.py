import torch
import torch.nn as nn
from losses.LossWithMask import LossWithMask

class ROLELoss(LossWithMask):
    def __init__(self, lamb=1, k=1, L=100):
        super(ROLELoss, self).__init__()
        self.epl = ExpectedPositiveLoss(lamb, k, L)

    def role_loss(self, F, Y, mask, labels):
        Y = Y.detach()
        epl_loss = self.epl(F, labels, mask)    
        orig_loss = nn.functional.binary_cross_entropy_with_logits(F, torch.sigmoid(Y), reduction='none')
        return epl_loss + orig_loss

    def forward(self, x, y, mask, pseudo=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        if pseudo is None: # For evaluation phase
            return nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        return 0.5 * (self.role_loss(x, pseudo, mask, y) + self.role_loss(pseudo, x, mask, y))


class ExpectedPositiveLoss(LossWithMask):
    def __init__(self, lamb=1, k=1, L=100):
        super(ExpectedPositiveLoss, self).__init__()
        self.lamb = lamb
        self.k = k
        self.L = L

    def forward(self, x, y, mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        probs = torch.sigmoid(x)
        avg_detections = torch.sum(probs) / len(probs) # Should be avg number per sample, len(probs) = batch len
        orig_loss = nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        orig_loss[y == 0] = 0
        return orig_loss + (self.lamb * ((avg_detections - self.k) / (self.L)) ** 2) / len(orig_loss.flatten())
