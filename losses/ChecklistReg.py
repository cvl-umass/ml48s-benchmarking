import torch
import torch.nn as nn
import numpy as np
from losses.LossWithMask import LossWithMask

class ChecklistReg(LossWithMask):
    def __init__(self, dataset, alpha=0, eps=0.001, init_val=None):
        super(ChecklistReg, self).__init__()
        self.dataset = dataset
        self.alpha = alpha
        self.eps = eps

        if init_val is not None:
            self.running_avg_pred = torch.ones((10000, 100)) * init_val
        else:
            self.running_avg_pred = torch.rand((10000, 100)) * (0.2) + 0.4 # Uniform(0.4, 0.6) like ROLE

        self.idx_to_asset_idx = torch.zeros(len(self.dataset)).long()
        asset = 0
        for i in range(len(dataset)):
            self.idx_to_asset_idx[i] = int(self.dataset.image_ids[i].split('/')[1])
            if self.idx_to_asset_idx[i] == asset:
                self.running_avg_pred[asset, np.where(np.logical_and(self.dataset.label_matrix[i] == 1, self.dataset.mask[i] == 1))] = 1
                asset += 1
    
    def forward(self, x, idx):
        asset_idx = self.idx_to_asset_idx[idx] # from idx to corresponding asset index
        probs = torch.sigmoid(x)
        
        prev_avg = self.running_avg_pred[asset_idx].to(x.device)
        reg_loss = self.alpha * nn.functional.binary_cross_entropy_with_logits(x, prev_avg, reduction='mean')
        
        with torch.no_grad():
            self.running_avg_pred[asset_idx] = self.running_avg_pred[asset_idx] * (1-self.eps) + probs.cpu() * self.eps
            self.running_avg_pred = self.running_avg_pred.detach()

        return reg_loss
