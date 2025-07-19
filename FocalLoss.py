import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率 p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 使用示例
criterion = FocalLoss(alpha=0.25, gamma=2.0)
input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
target = torch.tensor([0, 1])
loss = criterion(input, target)
print(loss)