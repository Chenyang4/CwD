# -*- coding: utf-8 -*-

"""
Pytorch port of the resnet used for CIFAR100 by iCaRL.

credits:
    https://github.com/arthurdouillard/incremental_learning.pytorch
    https://github.com/srebuffi/iCaRL/blob/master/iCaRL-TheanoLasagne/utils_cifar100.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DownsampleStride(nn.Module):
    def __init__(self, n=2):
        super(DownsampleStride, self).__init__()
        self._n = n

    def forward(self, x):
        return x[..., ::2, ::2]


class DownsampleConv(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, stride=2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        increase_dim=False,
        last_relu=False,
        downsampling="stride",
    ):
        super(ResidualBlock, self).__init__()

        self.increase_dim = increase_dim

        if increase_dim:
            first_stride = 2
            planes = inplanes * 2
        else:
            first_stride = 1
            planes = inplanes

        self.conv_a = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=first_stride,
            padding=1,
            bias=False,
        )
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_b = nn.BatchNorm2d(planes)

        if increase_dim:
            if downsampling == "stride":
                self.downsampler = DownsampleStride()
                self._need_pad = True
            else:
                self.downsampler = DownsampleConv(inplanes, planes)
                self._need_pad = False

        self.last_relu = last_relu

    @staticmethod
    def pad(x):
        return torch.cat((x, x.mul(0)), 1)

    def forward(self, x):
        y = self.conv_a(x)
        y = self.bn_a(y)
        y = F.relu(y, inplace=True)

        y = self.conv_b(y)
        y = self.bn_b(y)

        if self.increase_dim:
            x = self.downsampler(x)
            if self._need_pad:
                x = self.pad(x)

        y = x + y

        if self.last_relu:
            y = F.relu(y, inplace=True)

        return y


class PreActResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, increase_dim=False, last_relu=False):
        super().__init__()

        self.increase_dim = increase_dim

        if increase_dim:
            first_stride = 2
            planes = inplanes * 2
        else:
            first_stride = 1
            planes = inplanes

        self.bn_a = nn.BatchNorm2d(inplanes)
        self.conv_a = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=first_stride,
            padding=1,
            bias=False,
        )

        self.bn_b = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if increase_dim:
            self.downsample = DownsampleStride()
            self.pad = lambda x: torch.cat((x, x.mul(0)), 1)
        self.last_relu = last_relu

    def forward(self, x):
        y = self.bn_a(x)
        y = F.relu(y, inplace=True)
        y = self.conv_a(x)

        y = self.bn_b(y)
        y = F.relu(y, inplace=True)
        y = self.conv_b(y)

        if self.increase_dim:
            x = self.downsample(x)
            x = self.pad(x)

        y = x + y

        if self.last_relu:
            y = F.relu(y, inplace=True)

        return y


class Stage(nn.Module):
    def __init__(self, blocks, block_relu=False):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.block_relu = block_relu

    def forward(self, x):
        intermediary_features = []

        for b in self.blocks:
            x = b(x)
            intermediary_features.append(x)

            if self.block_relu:
                x = F.relu(x)

        return intermediary_features, x


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(
        self,
        n=5,
        nf=16,
        channels=3,
        preact=False,
        zero_residual=True,
        pooling_config={"type": "avg"},
        downsampling="stride",
        final_layer=False,
        all_attentions=False,
        last_relu=False,
        **kwargs
    ):
        """Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        if kwargs:
            raise ValueError("Unused kwargs: {}.".format(kwargs))

        self.all_attentions = all_attentions
        self._downsampling_type = downsampling
        self.last_relu = last_relu

        Block = ResidualBlock if not preact else PreActResidualBlock

        super(CifarResNet, self).__init__()

        self.conv_1_3x3 = nn.Conv2d(
            channels, nf, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_1 = nn.BatchNorm2d(nf)

        self.stage_1 = self._make_layer(Block, nf, increase_dim=False, n=n)
        self.stage_2 = self._make_layer(Block, nf, increase_dim=True, n=n - 1)
        self.stage_3 = self._make_layer(
            Block, 2 * nf, increase_dim=True, n=n - 2
        )
        self.stage_4 = Block(
            4 * nf,
            increase_dim=False,
            last_relu=False,
            downsampling=self._downsampling_type,
        )

        #  self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = 4 * nf
        if final_layer in (True, "conv"):
            self.final_layer = nn.Conv2d(
                self.out_dim, self.out_dim, kernel_size=1, bias=False
            )
        elif isinstance(final_layer, dict):
            if final_layer["type"] == "one_layer":
                self.final_layer = nn.Sequential(
                    nn.BatchNorm1d(self.out_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(
                        self.out_dim,
                        int(self.out_dim * final_layer["reduction_factor"]),
                    ),
                )
                self.out_dim = int(
                    self.out_dim * final_layer["reduction_factor"]
                )
            elif final_layer["type"] == "two_layers":
                self.final_layer = nn.Sequential(
                    nn.BatchNorm1d(self.out_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.out_dim, self.out_dim),
                    nn.BatchNorm1d(self.out_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(
                        self.out_dim,
                        int(self.out_dim * final_layer["reduction_factor"]),
                    ),
                )
                self.out_dim = int(
                    self.out_dim * final_layer["reduction_factor"]
                )
            else:
                raise ValueError(
                    "Unknown final layer type {}.".format(final_layer["type"])
                )
        else:
            self.final_layer = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )

        if zero_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn_b.weight, 0)

        self.num_features = self.out_dim

    def _make_layer(self, Block, planes, increase_dim=False, n=None):
        layers = []

        if increase_dim:
            layers.append(
                Block(
                    planes,
                    increase_dim=True,
                    last_relu=False,
                    downsampling=self._downsampling_type,
                )
            )
            planes = 2 * planes

        for i in range(n):
            layers.append(
                Block(
                    planes,
                    last_relu=False,
                    downsampling=self._downsampling_type,
                )
            )

        return Stage(layers, block_relu=self.last_relu)

    @property
    def last_conv(self):
        return self.stage_4.conv_b

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)

        feats_s1, x = self.stage_1(x)
        feats_s2, x = self.stage_2(x)
        feats_s3, x = self.stage_3(x)
        x = self.stage_4(x)

        return x

        """
        raw_features = self.end_features(x)
        features = self.end_features(F.relu(x, inplace=False))

        if self.all_attentions:
            attentions = [*feats_s1, *feats_s2, *feats_s3, x]
        else:
            attentions = [feats_s1[-1], feats_s2[-1], feats_s3[-1], x]

        return {
            "raw_features": raw_features,
            "features": features,
            "attention": attentions,
        }
        """

    def end_features(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if self.final_layer is not None:
            x = self.final_layer(x)

        return x


def resnet32(n=5, **kwargs):
    return CifarResNet(n=n, **kwargs)
