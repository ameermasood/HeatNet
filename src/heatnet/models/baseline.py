"""Baseline RGB heatmap regression model."""

import torch.nn as nn
import torchvision.models as models


class HeatmapHead(nn.Module):
    def __init__(self, num_keypoints, in_channels):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.deconv(x)
        x = self.final(x)
        return x


class KeypointHeatmapNet(nn.Module):
    def __init__(self, num_keypoints=50):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.head = HeatmapHead(num_keypoints, in_channels=512)

    def forward(self, x):
        feat = self.backbone(x)
        heatmaps = self.head(feat)
        return heatmaps
