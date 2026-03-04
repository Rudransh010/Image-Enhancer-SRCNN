# model.py - Edge-optimized lightweight SR (residual + depthwise-separable convs)
import torch
import torch.nn as nn
from typing import Optional


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise (groups=in_channels) then pointwise 1x1."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    """Lightweight residual block using depthwise conv in middle and 1x1 bottlenecks."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        # depthwise conv (keeps channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                            groups=channels, bias=False)
        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.dw(out)
        out = self.act2(out)
        out = self.conv2(out)
        return out + residual


class DSBlock(nn.Module):
    """DepthwiseSeparable block with a residual connection."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ds = DepthwiseSeparableConv(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ds(x)
        out = self.act(out)
        return out + x


class EdgeSRCNN(nn.Module):
    """
    Lightweight residual SR network.
    - Expectation: input `x` is the bicubic-upscaled image (RGB, float32, range whatever you train on).
    - Model learns residual to add to bicubic.
    """
    def __init__(self, num_channels: int = 48, num_ds_blocks: int = 3, num_res_blocks: int = 2) -> None:
        super().__init__()
        self.num_channels = num_channels

        # initial feature extraction
        self.conv_first = nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act_first = nn.ReLU(inplace=True)

        # efficient blocks
        self.ds_blocks = nn.Sequential(*[DSBlock(num_channels) for _ in range(num_ds_blocks)])
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # fusion after blocks (1x1 to mix channels)
        self.fusion = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)

        # final reconstruction to 3 channels (residual)
        self.conv_last = nn.Conv2d(num_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # initialization (optional but helpful)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input check
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("Input must be a 4D tensor with 3 channels (N,3,H,W). "
                             "Model expects bicubic-upscaled input.")

        bicubic = x  # caller should provide bicubic-upscaled LR -> HR sized input

        feat = self.conv_first(x)
        feat = self.act_first(feat)

        # keep global skip from early features
        feat_skip = feat

        feat = self.ds_blocks(feat)
        feat = self.res_blocks(feat)

        # global fusion and residual feature addition
        feat = self.fusion(feat)
        feat = feat + feat_skip

        residual = self.conv_last(feat)
        out = bicubic + residual
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(num_channels: int = 48, num_ds_blocks: int = 3, num_res_blocks: int = 2,
                 max_params: Optional[int] = 500_000) -> EdgeSRCNN:
    model = EdgeSRCNN(num_channels=num_channels, num_ds_blocks=num_ds_blocks, num_res_blocks=num_res_blocks)
    params = count_parameters(model)
    print(f"Parameter count: {params:,}")
    if max_params is not None and params > max_params:
        raise AssertionError(f"Model has {params:,} params which exceeds the {max_params:,} limit.")
    return model


def prepare_for_quantization(model: nn.Module) -> nn.Module:
    """
    Minimal guidance for static quantization:
    - Replace nn.ReLU with nn.ReLU (already used), ensure modules you want to fuse are in nn.Sequential
    - Typical fuses: ['conv', 'relu'] pairs; fuse conv+relu+conv patterns before torch.quantization.prepare
    This function does not mutate; it just returns the model for you to call quantization utilities on.
    """
    # Example (no-op here). Real preparation requires scripted fusing depending on your exact model layout.
    return model


if __name__ == "__main__":
    # quick sanity run (CPU) — safe to run in Colab / local
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model()  # adjust channels/blocks if you need different size
    model = model.to(device)
    model.eval()

    # create a dummy bicubic-upscaled input: batch 1, 3 channels, 256x256
    x = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")