import torch
import torch.nn as nn


class MultiLoss(nn.Module):
    def __init__(self, alpha=0.75, beta=0.1, gamma=2):
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def FocalLoss(self, x, y):
        x = torch.sigmoid(x)
        x_t = x * (2 * y - 1) + (1 - y)

        alpha_t = torch.ones_like(x_t) * self.alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)

        loss = -alpha_t * (1 - x_t) ** self.gamma * x_t.log()
        return loss.mean()

    def forward(self, predict, label):
        cls_pre = predict[..., 0]
        cls_lab = label[..., 0]
        cls_loss = self.FocalLoss(cls_pre, cls_lab)

        reg_pre = predict[..., 1:]
        reg_lab = label[..., 1:]

        positive_mask = torch.nonzero(torch.sum(torch.abs(reg_lab), dim=1))
        pos_regression_label = reg_lab[positive_mask.squeeze(), :]
        pos_regression_prediction = reg_pre[positive_mask.squeeze(), :]

        smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        reg_loss = smooth_l1(reg_pre, reg_lab) * self.beta

        multi_loss = cls_loss.add(reg_loss)

        return multi_loss

def test():
    loss = MultiLoss()
    pred = torch.sigmoid(torch.randn(1, 2, 2, 3))
    label = torch.tensor([[[[1, 0.4, 0.5], [0, 0.2, 0.5]], [[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    loss = loss(pred, label)
    print(loss)
test()