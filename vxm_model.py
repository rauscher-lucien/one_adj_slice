import torch
import torch.nn as nn
import torch.nn.functional as F


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

class NewUNetVXM(nn.Module):
    def __init__(self):
        super(NewUNetVXM, self).__init__()
        self.base = 32

        # Define a simpler network with fewer layers
        self.down0 = DownBlock(in_ch=2, out_ch=self.base)
        self.down1 = DownBlock(in_ch=self.base, out_ch=2*self.base)
        # Remove one down block and corresponding up block to make it less deep
        self.conv = ConvBlock(in_ch=2*self.base, out_ch=2*self.base)
        self.up1 = UpBlock(in_ch=2*self.base, out_ch=self.base)
        self.up0 = UpBlock(in_ch=self.base, out_ch=self.base)
        self.outconv = nn.Conv2d(self.base, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x0_a, x0_b = self.down0(x)
        x1_a, x1_b = self.down1(x0_b)
        x2 = self.conv(x1_b)
        x1 = self.up1(x2, x1_a)
        x0 = self.up0(x1, x0_a)
        x = self.outconv(x0)
        return x



class SpatialTransformer2D(nn.Module):
    def __init__(self, inshape):
        super().__init__()
        meshgrid = torch.meshgrid([torch.arange(0, s) for s in inshape])
        self.register_buffer('grid', torch.stack(meshgrid, dim=0).float())
    
    def forward(self, src, flow):
        new_locs = self.grid + flow
        for i in range(len(flow.shape) - 1):
            new_locs[i] = 2 * (new_locs[i] / (src.shape[i + 2] - 1) - 0.5)
        new_locs = new_locs.permute(1, 2, 0).unsqueeze(0)
        return F.grid_sample(src, new_locs, align_corners=True)

class VxmDense2D(nn.Module):
    def __init__(self, inshape, nb_features=[64, 128, 256, 512]):
        super().__init__()
        self.unet = NewUNetVXM()
        self.flow_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.transformer = SpatialTransformer2D(inshape)
    
    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.unet(x)
        flow = self.flow_conv(x)
        warped_src = self.transformer(src, flow)
        warped_tgt = self.transformer(tgt, flow)
        return warped_src, warped_tgt, flow


