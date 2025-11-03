# UNet Segmentation Model

This project includes a PyTorch implementation of the UNet architecture for semantic segmentation tasks. The implementation lives in `src/models/unet.py` and follows the classic encoder-decoder design with skip connections.

## Key Features
- Configurable depth and number of feature channels via the `features` argument.
- Choice between bilinear upsampling or learned transposed convolutions for the decoder.
- Optional batch normalization on every convolutional block.
- He initialization for convolution weights for stable training out of the box.

## Usage

```python
import torch
from src.models.unet import UNet

# Instantiate the model
model = UNet(
    in_channels=3,
    num_classes=2,
    features=[64, 128, 256, 512, 1024],
    bilinear=True,
    batch_norm=True,
)

# Forward pass
inputs = torch.randn(1, 3, 256, 256)
logits = model(inputs)
print(logits.shape)  # torch.Size([1, 2, 256, 256])
```

## Customisation Tips
- **Input channels**: adjust `in_channels` when working with grayscale (`1`) or multi-spectral imagery.
- **Number of classes**: set `num_classes` to match your dataset labels; the model outputs raw logits suitable for `torch.nn.CrossEntropyLoss`.
- **Decoder upsampling**: set `bilinear=False` to switch to learned transposed convolutions if you observe checkerboard artefacts.
- **Feature widths**: supply a custom `features` list (length ? 2) to control capacity. The list should include the output channels for each encoder stage, ending with the bottleneck width.

## Training Notes
- Pair the logits with `torch.nn.CrossEntropyLoss` for multi-class segmentation or `torch.nn.BCEWithLogitsLoss` for binary masks.
- Apply data augmentations such as random flips and elastic deformations to improve generalisation.
- Monitor metrics like Intersection-over-Union (IoU) or Dice coefficient for model evaluation.
