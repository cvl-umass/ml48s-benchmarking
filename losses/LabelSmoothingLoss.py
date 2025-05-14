import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        probs = torch.sigmoid(x)
        loss_mtx = torch.zeros_like(y).float()
        loss_mtx[y == 1] = (1.0 - self.eps / 2) * -torch.log((probs[y == 1])) + self.eps / 2 * -torch.log(1.0 - probs[y == 1])
        loss_mtx[y != 1] = (1.0 - self.eps / 2) * -torch.log(1.0 - probs[y != 1]) + self.eps / 2 * -torch.log(probs[y != 1])
        return loss_mtx
