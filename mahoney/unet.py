import numpy as np

import torch
import torch.nn as N
import torch.nn.functional as F


def lookup_activation(activation, **kwargs):
    '''Lookup an activation function by name.

    If activation is already a function, it is returned back. Otherwise it is
    cast to a str and used to look up an attribute of `torch.nn.functional`.

    If kwargs are given, they are bound to the function.
    '''
    if not callable(activation):
        activation = getattr(F, str(activation))
    return lambda x: activation(x, **kwargs)


class Conv2d(N.module):
    '''A 2D Convolution with support for fractional stride.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super().__init__()
        assert 0 < stride
        if stride < 1:
            stride = np.round(1/stride)
            conv =  N.ConvTranspose2d
        else:
            conv = N.Conv2d
        self.conv = conv(in_channels, out_channels, kernel_size, stride, **kwargs)

    def forward(self, x):
        return self.conv(x)


class VggBlock(N.module):
    '''A VGG-like block of convolution layers.
    '''
    def __init__(self, *channels, activation='relu', **kwargs):
        super().__init__()
        kwargs.set_default('kernel_size', 3)
        kwargs.set_default('padding', 1)
        kwargs.set_default('stride', 1)
        stride = kwargs.pop(stride)  # only applies to final layer

        self.activation = lookup_activation(activation)
        self.convs = []

        n_layers = len(channels) - 1
        for i in range(n_layers):
            s = stride if i+1 == n_layers else 1
            conv = Conv2d(channels[i], channels[i+1], stride=s, **kwargs)
            self.convs.append(conv)

    def forward(self, x):
        for conv in self.convs:
            x = self.activation(conv(x))
        return x


class UNet(N.module):
    '''A fully convolutional network based on U-Net.
    '''
    def __init__(self, n_channels, n_classes, block=VggBlock):
        super().__init__()

        self.cat = lambda a, b: torch.cat((a, b), 1)

        self.down1 = block(n_channels, 64, 64, 64, stride=2)
        self.down2 = block(64, 128, 128, 128, stride=2)
        self.down3 = block(128, 256, 256, 256, stride=2)
        self.down4 = block(256, 512, 512, 512, stride=2)

        self.bottom = block(512, 1024, 1024, 512, stride=1/2)

        self.up4 = block(512+512, 512, 512, 256, stride=1/2)
        self.up3 = block(256+256, 256, 256, 128, stride=1/2)
        self.up2 = block(128+128, 128, 128, 64, stride=1/2)
        self.up1 = block(64+64, 64, 64, stride=1)

        self.output = N.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        bottom = self.bottom(down4)
        up4 = self.up4(self.cat(bottom, down4))
        up3 = self.up3(self.cat(up4, down3))
        up2 = self.up2(self.cat(up3, down2))
        up1 = self.up1(self.cat(up2, down1))
        h = self.output(up1)
        return h
