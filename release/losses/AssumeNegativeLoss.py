import torch
import torch.nn as nn
from losses.LossWithMask import LossWithMask

# Just BCE

class AssumeNegativeLoss(LossWithMask):
    def __init__(self):
        super(AssumeNegativeLoss, self).__init__()

    def forward(self, x, y, mask):
        probs = torch.sigmoid(x)
        return -(y * torch.log(probs + 1e-5) + (1 - y) * torch.log(1 - probs + 1e-5))
