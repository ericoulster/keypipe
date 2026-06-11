"""
KeyNet CNN architecture for musical key detection.

Based on Korzeniowski & Widmer (2018): "Genre-Agnostic Key Classification
With Convolutional Neural Networks"

This architecture is designed to generalize across musical genres by using
only convolutional and pooling layers, omitting dense layers to reduce
overfitting and allow for deeper, more expressive models.
"""

import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    """
    Basic convolutional block: Conv2D -> BatchNorm -> ELU

    Each block consists of a 2D convolution, followed by batch normalization
    and ELU activation. This design allows for stable and efficient learning
    on log-magnitude log-frequency audio spectrograms.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output feature maps
        kernel_size: Size of the convolutional kernels
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding='same',
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        return x


class KeyNet(nn.Module):
    """
    Convolutional neural network for musical key classification.

    The network operates directly on log-frequency spectrogram snippets
    and outputs predicted key class logits.

    Args:
        num_classes: Number of key classes (default: 24 for 12 major + 12 minor)
        in_channels: Input feature channels (default: 1 for mono spectrograms)
        Nf: Number of feature maps for first convolution (controls model width)
        p: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 24,
        in_channels: int = 1,
        Nf: int = 20,
        p: float = 0.5
    ):
        super().__init__()

        # Block 1: Initial convolutions with larger kernel for spectral context
        self.conv1 = BasicConv2d(in_channels, Nf, kernel_size=5)
        self.conv2 = BasicConv2d(Nf, Nf, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(p=p)

        # Block 2: Increased feature maps
        self.conv3 = BasicConv2d(Nf, 2*Nf, kernel_size=3)
        self.conv4 = BasicConv2d(2*Nf, 2*Nf, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(p=p)

        # Block 3: Further channel doubling
        self.conv5 = BasicConv2d(2*Nf, 4*Nf, kernel_size=3)
        self.conv6 = BasicConv2d(4*Nf, 4*Nf, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(p=p)

        # Block 4: Deep layers for complex pattern extraction
        self.conv7 = BasicConv2d(4*Nf, 8*Nf, kernel_size=3)
        self.dropout4 = nn.Dropout2d(p=p)
        self.conv8 = BasicConv2d(8*Nf, 8*Nf, kernel_size=3)
        self.dropout5 = nn.Dropout2d(p=p)

        # Final classifier: 1x1 convolution (no dense layer)
        self.conv9 = BasicConv2d(8*Nf, num_classes, kernel_size=1)

        # Global average pooling for fixed-size output
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, channels, freq_bins, time_frames)
               Typically (B, 1, 104, T) for CQT spectrograms

        Returns:
            Logits tensor (batch, num_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = self.conv7(x)
        x = self.dropout4(x)
        x = self.conv8(x)
        x = self.dropout5(x)

        # Classifier
        x = self.conv9(x)
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)

        return x


def load_model(
    model_path: str,
    device: torch.device,
    num_classes: int = 24,
    in_channels: int = 1,
    Nf: int = 20
) -> KeyNet:
    """
    Load a pretrained KeyNet model from disk.

    Args:
        model_path: Path to the saved model weights (.pt file)
        device: Target device (CPU or CUDA)
        num_classes: Number of key classes
        in_channels: Input channels
        Nf: Number of feature maps in first convolution

    Returns:
        Loaded model ready for inference
    """
    model = KeyNet(
        num_classes=num_classes,
        in_channels=in_channels,
        Nf=Nf
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model
