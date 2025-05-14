import torch
import torch.nn as nn
import math

# https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Large_Loss_Matters_in_Weakly_Supervised_Multi-Label_Classification_CVPR_2022_paper.pdf

class IgnoreLargeLoss(nn.Module):
    """Use for target-only! Assumes all negative samples are unnanotated"""
    def __init__(self, mode='rejection', delta=0.2):
        super(IgnoreLargeLoss, self).__init__()
        self.delta = delta
        self.epoch = 1
        self.mode = mode
        assert self.mode in ['rejection', 'temp_correction', 'perm_correction']

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, x, y, mask):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        orig_loss = nn.functional.binary_cross_entropy_with_logits(x, y, reduction='none')
        if self.mode == 'perm_correction':
            frac = self.delta
        else:
            frac = (self.delta * (self.epoch - 1))
        k = math.ceil(x.shape[0] * x.shape[1] * frac)

        loss_unobs = (mask == 0) * orig_loss
        out_loss = orig_loss

        if k > 0:
            topk_loss_value = torch.topk(loss_unobs.flatten(), k).values[-1]
            if self.mode == 'rejection':
                out_loss[mask == 0] = torch.where(loss_unobs < topk_loss_value, orig_loss, torch.zeros_like(orig_loss))[mask == 0]
            elif self.mode == 'temp_correction' or 'perm_correction':
                corrected_loss = nn.functional.binary_cross_entropy_with_logits(x, torch.logical_not(y).float(), reduction='none')
                out_loss[mask == 0] = torch.where(loss_unobs < topk_loss_value, orig_loss, corrected_loss)[mask == 0]
        if self.mode == 'perm_correction':
            return out_loss, torch.where(torch.logical_and(loss_unobs >= topk_loss_value, mask == 0))
        else:
            return out_loss
