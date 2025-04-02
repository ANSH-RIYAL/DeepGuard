import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

class MappingNetwork(nn.Module):
    """Mapping network that transforms random latent vectors into style vectors."""
    def __init__(self, latent_dim: int, style_dim: int, n_layers: int = 8):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.extend([
                nn.Linear(latent_dim if i == 0 else style_dim, style_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.net = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class ModulatedConv2d(nn.Module):
    """Modulated convolution layer for style-based generation."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
        downsample: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        
        self.scale = 1 / np.sqrt(in_channels * kernel_size ** 2)
        self.padding = kernel_size // 2
        
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.modulation = nn.Linear(style_dim, in_channels)
        
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Modulate
        style = self.modulation(style).view(batch_size, 1, -1, 1, 1)
        weight = self.scale * self.weight * style
        
        # Demodulate
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)
        
        # Reshape weight for group convolution
        weight = weight.view(batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        # Reshape input
        x = x.view(1, batch_size * self.in_channels, x.shape[-2], x.shape[-1])
        
        # Convolve
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif self.downsample:
            x = F.avg_pool2d(x, 2)
        
        # Apply convolution
        x = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
        
        # Reshape output
        x = x.view(batch_size, self.out_channels, x.shape[-2], x.shape[-1])
        
        return x

class TemporalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, frames, height, width)
        batch, channels, frames, height, width = x.size()
        
        # Reshape for attention
        q = self.query(x.view(batch * frames, channels, height, width))
        k = self.key(x.view(batch * frames, channels, height, width))
        v = self.value(x.view(batch * frames, channels, height, width))
        
        # Compute attention
        q = q.view(batch, frames, channels, -1).permute(0, 2, 1, 3)
        k = k.view(batch, frames, channels, -1).permute(0, 2, 3, 1)
        v = v.view(batch, frames, channels, -1).permute(0, 2, 1, 3)
        
        attention = torch.bmm(q, k) * (1.0 / (channels ** 0.5))
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(attention, v)
        out = out.permute(0, 2, 1, 3).view(batch, frames, channels, height, width)
        out = out.permute(0, 2, 1, 3, 4)
        
        return self.gamma * out + x

class Generator(nn.Module):
    """Style-based generator for high-quality image/video generation."""
    def __init__(
        self,
        latent_dim: int = 512,
        style_dim: int = 512,
        num_channels: int = 3,
        hidden_channels: List[int] = [512, 512, 256, 128, 64],
        frame_size: Tuple[int, int] = (256, 256),
        num_frames: int = 16
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels
        self.frame_size = frame_size
        self.num_frames = num_frames
        
        # Mapping network
        self.mapping = MappingNetwork(latent_dim, style_dim)
        
        # Initial constant input
        self.constant = nn.Parameter(torch.randn(1, hidden_channels[0], 8, 8))
        
        # Style convolutions
        self.conv1 = ModulatedConv2d(hidden_channels[0], hidden_channels[1], 3, style_dim, upsample=True)
        self.conv2 = ModulatedConv2d(hidden_channels[1], hidden_channels[2], 3, style_dim, upsample=True)
        self.conv3 = ModulatedConv2d(hidden_channels[2], hidden_channels[3], 3, style_dim, upsample=True)
        self.conv4 = ModulatedConv2d(hidden_channels[3], hidden_channels[4], 3, style_dim, upsample=True)
        self.conv5 = ModulatedConv2d(hidden_channels[4], num_channels * num_frames, 3, style_dim, upsample=True)
        
        # Activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        
        # Generate styles
        styles = self.mapping(z)
        
        # Initial constant input
        x = self.constant.repeat(batch_size, 1, 1, 1)  # [B, C, H, W]
        
        # Apply style convolutions
        x = self.conv1(x, styles)  # [B, C, H*2, W*2]
        x = self.activation(x)
        
        x = self.conv2(x, styles)  # [B, C, H*4, W*4]
        x = self.activation(x)
        
        x = self.conv3(x, styles)  # [B, C, H*8, W*8]
        x = self.activation(x)
        
        x = self.conv4(x, styles)  # [B, C, H*16, W*16]
        x = self.activation(x)
        
        x = self.conv5(x, styles)  # [B, T*C, H*32, W*32]
        x = torch.tanh(x)  # Output in range [-1, 1]
        
        return x

class Discriminator(nn.Module):
    """Style-based discriminator for high-quality image/video discrimination."""
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: List[int] = [64, 128, 256, 512, 512],
        frame_size: Tuple[int, int] = (256, 256),
        num_frames: int = 16
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.num_channels = num_channels
        
        # Process frames with 2D convolutions
        self.conv1 = nn.Conv2d(num_channels * num_frames, hidden_channels[0], 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2], 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels[2], hidden_channels[3], 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(hidden_channels[3], hidden_channels[4], 3, stride=2, padding=1)
        
        # Calculate final output size
        h = frame_size[0] // 32
        w = frame_size[1] // 32
        self.final_size = hidden_channels[4] * h * w
        
        self.fc = nn.Linear(self.final_size, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, T*C, H, W]
        batch_size = x.size(0)
        
        # Apply convolutions
        x = self.activation(self.conv1(x))  # [B, H0, H/2, W/2]
        x = self.activation(self.conv2(x))  # [B, H1, H/4, W/4]
        x = self.activation(self.conv3(x))  # [B, H2, H/8, W/8]
        x = self.activation(self.conv4(x))  # [B, H3, H/16, W/16]
        x = self.activation(self.conv5(x))  # [B, H4, H/32, W/32]
        
        # Flatten and apply final linear layer
        x = x.view(batch_size, -1)  # [B, H4*H/32*W/32]
        x = self.fc(x)
        
        return x 