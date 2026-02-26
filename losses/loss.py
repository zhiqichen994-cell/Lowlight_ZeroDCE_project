import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, output, target):
        # 1. 解决 tuple 报错：如果 output 是元组/列表，取最后一个（通常是最终增强图）
        if isinstance(output, (list, tuple)):
            output = output[-1]

        # 2. 形状检查与处理
        # Zero-DCE 如果输出通道是 24 (8*3)，取最后三通道作为最终结果
        if output.shape[1] > 3:
            output = output[:, -3:, :, :]

        # 3. 尺寸对齐：确保高宽一致
        if output.shape[2:] != target.shape[2:]:
            output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)

        return self.loss(output, target)