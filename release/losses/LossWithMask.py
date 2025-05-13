import torch
import torch.nn as nn

class LossWithMask(nn.Module):
    """Multilabel with logits, predictions, mask for whether the data is annotated"""
    def __init__(self):
        super(LossWithMask, self).__init__()

    def forward(self, logits, targets, mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        mask: annotated status of each label
        """
        raise NotImplementedError
