import torch
import torch.nn as nn


class MultiLoss(nn.Module):
    def __init__(self, alpha=0.75, beta=0.1, gamma=2):
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def FocalLoss(self, x, y):
        pt = torch.where(y == 1.0, x, 1 - x)
        loss = -1 * (1 - pt) ** self.gamma * torch.log(pt)
        return loss.mean()

    def forward(self, predict, label):
        cls_pre = predict[..., -1]
        cls_lab = label[..., -1]
        cls_loss = self.FocalLoss(cls_pre, cls_lab)

        reg_pre = predict[..., :-1]
        reg_lab = label[..., :-1]

        smooth_l1 = nn.SmoothL1Loss(reduction='mean', beta=self.beta)
        reg_loss = smooth_l1(reg_pre, reg_lab)
        multi_loss = cls_loss.add(reg_loss)
        return multi_loss