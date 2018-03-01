import numpy as np

import torch
import torch.nn as N
import torch.nn.functional as F


class VggBlock(N.Module):
    '''A VGG-like block of convolution layers.

    A VggBlock is a series of convolutions optionally followed by some
    resampling layer. Each convolution uses a 3x3 kernel with a stride 1
    and padding to ensure each output maintains spatial dimension.

    After the main convolutions, the output may be resampled. The `resample`
    argument can be a torch Module to use for resampling, a function which
    accepts a channel size and returns a torch Module, or one of a string
    specifying a known resample operation.
    '''

    def __init__(self, in_channel, *channels, resample=None):
        '''Construct a VggBlock given a list of channel dimensions.

        Args:
            in_channel:
                The number of channels in the input.
            channels:
                A list of output channels, one for each conv layer.
            resample:
                The resample operation. It may be a torch Module, a function
                which takes the channel size and returns a torch Module, or one
                of the following strings:
                - 'max': max pooling with stride 2.
                - 'conv': 2x2 convolution with stride 2.
                - 'deconv': 2x2 convolution with fractional stride 1/2.
        '''
        super().__init__()
        layers = []

        # The main layers
        c0 = in_channel
        for c1 in range(channels):
            conv = N.Conv2d(c0, c1, kernel_size=3, padding=1)
            relu = N.ReLU(inplace=True)
            layers += [conv, relu]
            c0 = c1

        # The resample layer.
        # Note that c0 == c1 at this point.
        if resample is not None:
            if resample == 'max': resample = N.MaxPool2d(kernel_size=2)
            if resample == 'conv': resample = N.Conv2d(c0, c1, kernel_size=2, stride=2)
            if resample == 'deconv': resample = N.ConvTranspose2d(c0, c1, kernel_size=2, stride=2)
            if callable(resample): resample = resample(c1)
            layers.append(resample)

        self.layers = N.Sequential(*layers)

    def forward(self, x):
        return self.layers.forward(x)


class UNet(N.Module):
    '''A fully convolutional network based on U-Net.
    '''
    def __init__(self, n_channels, n_classes, block=VggBlock):
        super().__init__()

        self.down1 = block(n_channels, 64, 64, resample='max')
        self.down2 = block(64, 128, 128, resample='max')
        self.down3 = block(128, 256, 256, resample='max')
        self.down4 = block(256, 512, 512, resample='max')

        self.bottom = block(512, 1024, 512, resample='deconv')

        self.up4 = block(512+512, 512, 256, resample='deconv')
        self.up3 = block(256+256, 256, 128, resample='deconv')
        self.up2 = block(128+128, 128, 64, resample='deconv')
        self.up1 = block(64+64, 64, resample=None)

        self.output = N.Conv2d(64, n_classes, kernel_size=1)

    def cat(self, a, b):
        return torch.cat((a, b), 1)

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
