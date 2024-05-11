import torch
import torch.nn as nn

class SimpleConvBlock(nn.Module):
    """A simple convolutional block: Conv2d -> BatchNorm2d -> LeakyReLU."""
    def __init__(self, in_channels, out_channels):
        super(SimpleConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)

class SimpleDownBlock(nn.Module):
    """A downsampling block with convolution and max pooling."""
    def __init__(self, in_channels, out_channels):
        super(SimpleDownBlock, self).__init__()
        self.convblock = SimpleConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.convblock(x)
        return self.pool(x)

class SimpleUpBlock(nn.Module):
    """An upsampling block using transposed convolution and then convolution."""
    def __init__(self, in_channels, out_channels):
        super(SimpleUpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.convblock = SimpleConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.upsample(x)
        return self.convblock(x)

class SimpleUNet(nn.Module):
    """A very simple U-Net without skip connections."""
    def __init__(self):
        super(SimpleUNet, self).__init__()
        base = 16  # Base channel count

        # Down path
        self.down0 = SimpleDownBlock(in_channels=1, out_channels=base)
        self.down1 = SimpleDownBlock(in_channels=base, out_channels=2 * base)

        # Bottom layer
        self.conv = SimpleConvBlock(in_channels=2 * base, out_channels=2 * base)

        # Up path
        self.up1 = SimpleUpBlock(in_channels=2 * base, out_channels=base)
        self.up0 = SimpleUpBlock(in_channels=base, out_channels=1)

        # Final output layer
        self.outconv0 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Down path
        x0 = self.down0(x)
        x1 = self.down1(x0)

        # Bottom layer
        x_c = self.conv(x1)

        # Up path
        x1 = self.up1(x_c)
        x0 = self.up0(x1)

        x = self.outconv0(x0)

        return x