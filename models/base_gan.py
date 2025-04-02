import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class MappingNetwork(nn.Module):
    """Mapping network that transforms random latent vectors into style vectors."""
    def __init__(self, latent_dim: int = 512, style_dim: int = 512, num_layers: int = 8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(latent_dim if i == 0 else style_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class ModulatedConv2d(nn.Module):
    """Modulated convolution layer for style-based generation."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, style_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.style_dim = style_dim
        
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.modulation = nn.Linear(style_dim, in_channels)
        self.demodulation = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        style = self.modulation(style).view(batch_size, 1, -1, 1, 1)
        weight = self.weight * style
        
        demod = self.demodulation.view(1, -1, 1, 1)
        weight = weight * demod
        
        x = F.conv2d(x, weight.view(-1, self.in_channels, self.kernel_size, self.kernel_size), 
                    padding=self.kernel_size//2, groups=batch_size)
        return x

class Generator(nn.Module):
    """Style-based generator for high-quality image/video generation."""
    def __init__(self, 
                 latent_dim: int = 512,
                 style_dim: int = 512,
                 channels: int = 3,
                 max_resolution: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.channels = channels
        self.max_resolution = max_resolution
        
        self.mapping = MappingNetwork(latent_dim, style_dim)
        
        # Initial constant input
        self.const_input = nn.Parameter(torch.randn(1, channels, 4, 4))
        
        # Progressive growing layers
        self.conv1 = ModulatedConv2d(channels, 512, 3, style_dim)
        self.conv2 = ModulatedConv2d(512, 512, 3, style_dim)
        self.conv3 = ModulatedConv2d(512, 256, 3, style_dim)
        self.conv4 = ModulatedConv2d(256, 128, 3, style_dim)
        self.conv5 = ModulatedConv2d(128, 64, 3, style_dim)
        self.conv6 = ModulatedConv2d(64, channels, 3, style_dim)
        
        self.noise_weights = nn.Parameter(torch.zeros(6))
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        styles = self.mapping(z)
        batch_size = z.shape[0]
        
        x = self.const_input.expand(batch_size, -1, -1, -1)
        
        # Progressive growing
        x = self.conv1(x, styles) + self.noise_weights[0] * torch.randn_like(x)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.conv2(x, styles) + self.noise_weights[1] * torch.randn_like(x)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.conv3(x, styles) + self.noise_weights[2] * torch.randn_like(x)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.conv4(x, styles) + self.noise_weights[3] * torch.randn_like(x)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.conv5(x, styles) + self.noise_weights[4] * torch.randn_like(x)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.conv6(x, styles) + self.noise_weights[5] * torch.randn_like(x)
        
        return torch.tanh(x)

class Discriminator(nn.Module):
    """Style-based discriminator for high-quality image/video discrimination."""
    def __init__(self, channels: int = 3, max_resolution: int = 1024):
        super().__init__()
        self.channels = channels
        
        # Progressive growing layers
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 1, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.avg_pool2d(x, 2)
        
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.avg_pool2d(x, 2)
        
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.avg_pool2d(x, 2)
        
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.avg_pool2d(x, 2)
        
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.avg_pool2d(x, 2)
        
        x = self.conv6(x)
        x = x.mean([2, 3])
        
        return x 