# model.py - Edge-optimized SRCNN with depthwise separable convolutions
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.prelu1 = nn.PReLU(channels)
        self.dwconv = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.prelu2 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.dwconv(out)
        out = self.prelu2(out)
        out = self.conv2(out)
        return out + residual


class DSBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dsconv = DepthwiseSeparableConv(channels, channels, 3, 1, 1)
        self.prelu = nn.PReLU(channels)
        
    def forward(self, x):
        residual = x
        out = self.dsconv(x)
        out = self.prelu(out)
        return out + residual


class EdgeSRCNN(nn.Module):
    def __init__(self, num_channels=48, num_ds_blocks=3, num_res_blocks=2):
        super().__init__()
        self.num_channels = num_channels
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(3, num_channels, 3, 1, 1, bias=True)
        self.prelu_first = nn.PReLU(num_channels)
        
        # Depthwise separable blocks
        self.ds_blocks = nn.ModuleList([DSBlock(num_channels) for _ in range(num_ds_blocks)])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # Final reconstruction
        self.conv_last = nn.Conv2d(num_channels, 3, 3, 1, 1, bias=True)
        
    def forward(self, x):
        # Bicubic upscale to target size (done outside or in preprocessing)
        # Model predicts residual to add to bicubic
        bicubic = x
        
        # Feature extraction
        feat = self.conv_first(x)
        feat = self.prelu_first(feat)
        
        # DS blocks
        for block in self.ds_blocks:
            feat = block(feat)
        
        # Residual blocks
        for block in self.res_blocks:
            feat = block(feat)
        
        # Reconstruction
        residual = self.conv_last(feat)
        out = bicubic + residual
        
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_channels=48, num_ds_blocks=3, num_res_blocks=2):
    model = EdgeSRCNN(num_channels, num_ds_blocks, num_res_blocks)
    params = count_parameters(model)
    print(f"Parameter count: {params:,}")
    assert params <= 500_000, f"Model has {params} parameters, exceeds 500k limit!"
    return model


if __name__ == "__main__":
    model = create_model()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")
