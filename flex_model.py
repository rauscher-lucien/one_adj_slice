import torch
import torch.nn as nn

from utils import *


class ConvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)
    

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_a =  self.convblock(x)
        x_b = self.pool(x_a)
        return x_a, x_b


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, padding=0)
        self.convblock = ConvBlock(2*in_ch, out_ch)

    def forward(self, x, x_new):
        x = self.upsample(x)
        x = torch.cat([x, x_new], dim=1)
        x = self.convblock(x)
        return x


class FlexibleUNet(nn.Module):
    def __init__(self, base_channels=32, in_channels=1, out_channels=1, depth=4):
        super(FlexibleUNet, self).__init__()
        self.base_channels = base_channels
        self.depth = depth

        # Initialize down-sampling layers dynamically
        self.down_layers = nn.ModuleList()
        self.down_layers.append(DownBlock(in_channels, base_channels))

        for i in range(1, depth):
            self.down_layers.append(DownBlock(base_channels * (2 ** (i - 1)), base_channels * (2 ** i)))

        # Central convolution block
        self.center_conv = ConvBlock(base_channels * (2 ** (depth - 1)), base_channels * (2 ** (depth - 1)))

        # Initialize up-sampling layers dynamically
        self.up_layers = nn.ModuleList()

        for i in range(depth - 1, 0, -1):
            self.up_layers.append(UpBlock(base_channels * (2 ** i), base_channels * (2 ** (i - 1))))

        self.up_layers.append(UpBlock(base_channels, base_channels))

        # Output layer
        self.outconv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Forward pass through down-sampling blocks
        down_outputs = []
        for down_layer in self.down_layers:
            x_a, x_b = down_layer(x)
            down_outputs.append(x_a)
            x = x_b
            #plot_intensity_line_distribution(x, 'down')

        # Central convolution block
        x = self.center_conv(x)

        # Forward pass through up-sampling blocks
        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x, down_outputs[-(i + 1)])
            #plot_intensity_line_distribution(x, 'up')

        # Final output layer
        x = self.outconv(x)
        return x
