"""UNet architecture for semantic segmentation tasks.

This module implements a configurable UNet model in PyTorch. It follows the
encoder-decoder design with skip connections introduced in the U-Net paper:
"U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al.

Example
-------
>>> import torch
>>> from src.models.unet import UNet
>>> model = UNet(in_channels=3, num_classes=2)
>>> x = torch.randn(1, 3, 256, 256)
>>> preds = model(x)
>>> preds.shape
torch.Size([1, 2, 256, 256])

"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_channels(candidates: Iterable[int]) -> List[int]:
    """Validate and normalize feature channel configuration."""

    channels = list(candidates)
    if len(channels) == 0:
        raise ValueError("`features` must contain at least one channel size")
    if any(c <= 0 for c in channels):
        raise ValueError("All feature sizes must be positive integers")
    return channels


class DoubleConv(nn.Sequential):
    """Two convolution layers each followed by batch norm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        *,
        batch_norm: bool = True,
    ) -> None:
        if mid_channels is None:
            mid_channels = out_channels

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=not batch_norm),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm)
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        super().__init__(*layers)


class Down(nn.Module):
    """Downscaling with maxpool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, *, batch_norm: bool = True) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, batch_norm=batch_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - short descr
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling step that merges encoder skip connections."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        bilinear: bool,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if in_channels % 2 != 0:
            raise ValueError("UNet decoder expects even number of input channels per up block")

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        up_channels = in_channels // 2
        self.conv = DoubleConv(up_channels + skip_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Pad x1 to match size of x2 when input dims are not powers of two
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution that maps features to segmentation logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - short descr
        return self.conv(x)


class UNet(nn.Module):
    """UNet segmentation model.

    Parameters
    ----------
    in_channels:
        Number of channels in the input image.
    num_classes:
        Number of output segmentation classes.
    features:
        Iterable specifying encoder feature sizes for each depth level.
        Defaults to the canonical [64, 128, 256, 512].
    bilinear:
        If ``True``, use bilinear upsampling in the decoder. Otherwise, use
        learned transposed convolutions.
    batch_norm:
        If ``True``, apply batch normalization after each convolution.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        features: Iterable[int] | None = None,
        bilinear: bool = True,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()

        feature_channels = _ensure_channels(features or [64, 128, 256, 512, 1024])
        if len(feature_channels) < 2:
            raise ValueError("`features` must provide at least two levels (encoder and bottleneck)")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.batch_norm = batch_norm

        self.inc = DoubleConv(in_channels, feature_channels[0], batch_norm=batch_norm)

        self.down_layers = nn.ModuleList()
        for in_f, out_f in zip(feature_channels, feature_channels[1:]):
            self.down_layers.append(Down(in_f, out_f, batch_norm=batch_norm))

        encoder_channels = feature_channels[:-1]
        bottleneck_channels = feature_channels[-1]

        self.bottleneck = DoubleConv(bottleneck_channels, bottleneck_channels, batch_norm=batch_norm)

        self.up_layers = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_ch in reversed(encoder_channels):
            self.up_layers.append(
                Up(
                    in_channels=current_channels,
                    skip_channels=skip_ch,
                    out_channels=skip_ch,
                    bilinear=bilinear,
                    batch_norm=batch_norm,
                )
            )
            current_channels = skip_ch

        self.outc = OutConv(current_channels, num_classes)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x_encoded = self.inc(x)
        skips.append(x_encoded)

        for down in self.down_layers[:-1]:
            x_encoded = down(x_encoded)
            skips.append(x_encoded)

        x_encoded = self.down_layers[-1](x_encoded)
        bottleneck = self.bottleneck(x_encoded)

        x_decoded = bottleneck
        for up_layer in self.up_layers:
            skip_connection = skips.pop()
            x_decoded = up_layer(x_decoded, skip_connection)

        logits = self.outc(x_decoded)
        return logits

    def reset_parameters(self) -> None:
        """He-initialize convolution layers for stable training."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


__all__ = ["UNet"]
