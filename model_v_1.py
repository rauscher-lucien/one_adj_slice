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

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, depth=4, use_skip_connections=True):
        super(UNet, self).__init__()
        self.depth = depth
        self.use_skip_connections = use_skip_connections

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        current_channels = in_channels
        for i in range(depth):
            next_channels = base_channels * (2 ** i)
            self.encoder_blocks.append(SimpleConvBlock(current_channels, next_channels))
            self.pooling_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = next_channels

        # Bottleneck
        self.bottleneck = SimpleConvBlock(current_channels, current_channels * 2)

        # Decoder
        self.upconv_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in reversed(range(depth)):
            next_channels = base_channels * (2 ** i)
            self.upconv_layers.append(nn.ConvTranspose2d(next_channels * 2, next_channels, kernel_size=2, stride=2))
            if use_skip_connections:
                self.decoder_blocks.append(SimpleConvBlock(next_channels * 2, next_channels))
            else:
                self.decoder_blocks.append(SimpleConvBlock(next_channels, next_channels))

        # Final output layer
        self.final_layer = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder pass
        skip_connections = []
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            if self.use_skip_connections:
                skip_connections.append(x)
            x = self.pooling_layers[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass
        for i in range(self.depth):
            x = self.upconv_layers[i](x)
            if self.use_skip_connections:
                skip = skip_connections[-(i + 1)]
                x = torch.cat((skip, x), dim=1)
            x = self.decoder_blocks[i](x)

        # Final layer
        return self.final_layer(x)

