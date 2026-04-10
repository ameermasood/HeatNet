"""Loss functions for HeatNet training."""

import torch


def focal_heatmap_loss(pred, gt, alpha=2.0, gamma=4.0, eps=1e-6):
    probabilities = torch.sigmoid(pred)
    positive = -alpha * (1 - probabilities) ** gamma * gt * torch.log(probabilities + eps)
    negative = -(1 - gt) * torch.log(1 - probabilities + eps)
    return (positive + negative).mean()
