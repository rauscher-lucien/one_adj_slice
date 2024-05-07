import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet2D(nn.Module):
    def __init__(self, infeats=2, nb_features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder
        for i, feats in enumerate(nb_features):
            in_channels = infeats if i == 0 else nb_features[i - 1]
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_channels, feats, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feats, feats, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ))
        
        # Decoder
        for i in reversed(range(1, len(nb_features))):
            feats = nb_features[i]
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(feats, feats // 2, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(feats, feats // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feats // 2, feats // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
        
        # Final layer
        self.final_conv = nn.Conv2d(nb_features[0], nb_features[0] // 2, kernel_size=3, padding=1)
        
    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        skips = skips[::-1]
        
        for i, dec in enumerate(self.decoders):
            x = dec(x)
            x = torch.cat((x, skips[i + 1]), dim=1)
        
        return self.final_conv(x)

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
        self.unet = Unet2D(infeats=2, nb_features=nb_features)
        self.flow_conv = nn.Conv2d(nb_features[0] // 2, 2, kernel_size=3, padding=1)
        self.transformer = SpatialTransformer2D(inshape)
    
    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        x = self.unet(x)
        flow = self.flow_conv(x)
        warped_src = self.transformer(src, flow)
        warped_tgt = self.transformer(tgt, flow)
        return warped_src, warped_tgt, flow

# Example usage:
# Define the model with an example 2D input shape
model = VxmDense2D(inshape=(128, 128))
source_image = torch.randn(1, 1, 128, 128)
target_image = torch.randn(1, 1, 128, 128)
warped, flow = model(source_image, target_image)
