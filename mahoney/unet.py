import numpy as np

import torch
import torch.nn as N
import torch.nn.functional as F


def _get_block(name):
    '''Maps string names to block classes.
    '''
    if name == 'vgg': return _Vgg
    raise Exception(f"Unknown block type '{name}'")


class _Vgg(N.Module):
    '''A simple block of 3x3 convolutions.
    '''
    def __init__(self, in_channel, *channels):
        super().__init__()
        self.layers = []
        c0 = in_channel
        for c1 in channels:
            conv = N.Conv2d(c0, c1, kernel_size=3, padding=1)
            relu = N.ReLU(inplace=True)
            self.layers += [conv, relu]
            c0 = c1

        # Initialize weights
        for m in self.layers:
            if not isinstance(m, N.BatchNorm2d):
                if hasattr(m, 'weight'): N.init.kaiming_uniform(m.weight)
                if hasattr(m, 'bias'): m.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _DownBlock(N.Module):
    '''A downward block for U-Net.
    '''
    def __init__(self, in_channel, *channels, block_type='vgg'):
        super().__init__()
        block = _get_block(block_type)
        self.conv = block(in_channel, *channels)
        self.resample = N.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        out = self.resample(skip)
        return out, skip


class _UpBlock(N.Module):
    '''An upward block for U-Net.
    '''
    def __init__(self, in_channel, *channels, block_type='vgg'):
        super().__init__()
        block = _get_block(block_type)
        self.conv = block(in_channel, *channels)
        self.resample = N.ConvTranspose2d(channels[-1], channels[-1]//2, kernel_size=2, stride=2)

    def forward(self, x, skip):
        conv = self.conv(x)
        up = self.resample(conv, output_size=skip.shape)
        out = torch.cat((up, skip), 1)
        return out


class UNet(N.Module):
    '''A fully convolutional network based on U-Net.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation".
    Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
    Medical Image Computing and Computer-Assisted Intervention (MICCAI),
    Springer, LNCS, Vol.9351: 234--241, 2015,
    https://arxiv.org/abs/1505.04597

    This architecture differs slightly from the original in that it includes
    an extra convolution on the input which makes it easier to have a flexible
    architecture.
    '''

    def __init__(self, n_channels, n_classes, depth=4, size=1024, block_type='vgg'):
        '''Construct a U-Net.

        Args:
            n_channels:
                The number of channels in the input.
            n_classes:
                The number of classes in the output.
            depth:
                The depth of the U, e.g. the number of down/up steps.
                The default of 4 is used in the original paper.
            size:
                The number of channels at the very bottom.
                The default of 1024 is used in the original paper.
            block_type:
                The type of convolution block. The default of 'vgg' is used in
                the original paper. The following are supported:
                - 'vgg': blocks of simple 3x3 convolutions with ReLU activation.
        '''
        super().__init__()
        block = _get_block(block_type)

        # Input
        c = size >> depth
        c0 = c // 2
        self.input = block(n_channels, c0)

        # Down
        self.down = []
        for i in range(depth):
            self.down.append(_DownBlock(c0, c, c, block_type=block_type))
            c0 = c
            c *= 2

        # Up
        self.up = []
        for i in range(depth):
            self.up.append(_UpBlock(c0, c, c, block_type=block_type))
            c0 = c
            c //= 2

        # Output
        self.top = block(c0, c, c)
        self.output = N.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x):
        # Input
        x = self.input(x)

        # Down
        skip = []
        for down in self.down:
            x, s = down(x)
            skip.append(s)

        # Up
        skip = list(reversed(skip))
        for s, up in zip(skip, self.up):
            x = up(x, s)

        # Output
        x = self.top(x)
        x = self.output(x)
        return x
