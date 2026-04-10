"""Cross-fusion RGB-D heatmap regression model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, channels, act_layer):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.05)
        self.act = act_layer
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.05)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.act(out)


class TwoStageFusion(nn.Module):
    def __init__(self, channels, act_layer):
        super().__init__()
        self.rb1_rgb = ResidualBlock(channels, act_layer)
        self.rb1_depth = ResidualBlock(channels, act_layer)
        self.rb2_rgb = ResidualBlock(channels, act_layer)
        self.rb2_depth = ResidualBlock(channels, act_layer)
        self.bn_fuse = nn.BatchNorm2d(2 * channels, momentum=0.05)

    def forward(self, x_rgb, x_depth):
        fbc = self.rb1_rgb(x_rgb)
        fbd = self.rb1_depth(x_depth)

        rgb_in = fbc + x_depth
        dpt_in = fbd + x_rgb

        out_rgb = self.rb2_rgb(rgb_in)
        out_dpt = self.rb2_depth(dpt_in)

        fused = torch.cat([out_rgb, out_dpt], dim=1)
        fused = self.bn_fuse(fused)
        return fused


class HeatmapHead(nn.Module):
    def __init__(self, num_keypoints, in_channels, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            act_layer,
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            act_layer,
            nn.Conv2d(256, num_keypoints, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return x


class CrossFuNet(nn.Module):
    def __init__(self, num_keypoints=50, act_layer=nn.Mish(inplace=True)):
        super().__init__()

        base_rgb = models.resnet18(pretrained=True)
        base_depth = models.resnet18(pretrained=True)

        base_rgb.layer3[0].conv1.stride = (1, 1)
        base_rgb.layer3[0].downsample[0].stride = (1, 1)
        base_depth.layer3[0].conv1.stride = (1, 1)
        base_depth.layer3[0].downsample[0].stride = (1, 1)

        self.rgb_encoder = nn.Sequential(*list(base_rgb.children())[:7])
        self.depth_encoder = nn.Sequential(*list(base_depth.children())[:7])
        self.depth_encoder[0] = nn.Conv2d(1, 64, 7, 2, 3, bias=False)

        self.cross_fusion = TwoStageFusion(256, act_layer)
        self.decoder = HeatmapHead(
            num_keypoints=num_keypoints, in_channels=512, act_layer=act_layer
        )

    def forward(self, rgb, depth):
        feat_rgb = self.rgb_encoder(rgb)
        feat_depth = self.depth_encoder(depth)

        assert feat_rgb.shape[1:] == feat_depth.shape[1:], (
            f"RGB vs Depth feature mismatch: {feat_rgb.shape} vs {feat_depth.shape}"
        )

        fused = self.cross_fusion(feat_rgb, feat_depth)

        assert fused.shape[-2:] == (32, 32) and fused.shape[1] == 512, (
            f"Fusion output wrong: {fused.shape}"
        )

        heatmaps = self.decoder(fused)

        assert heatmaps.shape[-2:] == (64, 64), (
            f"Head output wrong: {heatmaps.shape}"
        )

        return heatmaps
