import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from typing import Tuple, Dict, Optional, List, Union
import matplotlib.pyplot as plt
from pathlib import Path

# Check if MPS (Metal Performance Shaders) is available for M2 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class GANTrainer:
    """Trainer class for GAN models with M2 Mac optimizations."""
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        data_loader: DataLoader,
        latent_dim: int = 512,
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        log_dir: str = "logs",
        save_dir: str = "checkpoints",
        quality: str = "high"
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.data_loader = data_loader
        self.latent_dim = latent_dim
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.quality = quality
        
        # Create directories
        self.log_dir = Path(log_dir) / quality
        self.save_dir = Path(save_dir) / quality
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2)
        )
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Loss functions
        self.criterion = nn.BCEWithLogitsLoss()
        
    def compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_step(self, real_images: torch.Tensor) -> Dict[str, float]:
        """Perform one training step."""
        batch_size = real_images.size(0)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim).to(device)
        fake_images = self.generator(z)
        
        # Real images
        real_validity = self.discriminator(real_images)
        d_real_loss = -torch.mean(real_validity)
        
        # Fake images
        fake_validity = self.discriminator(fake_images.detach())
        d_fake_loss = torch.mean(fake_validity)
        
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_images, fake_images.detach())
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss + self.lambda_gp * gradient_penalty
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        if self.n_critic == 0 or (self.n_critic > 0 and self.n_critic_step % self.n_critic == 0):
            self.g_optimizer.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, self.latent_dim).to(device)
            fake_images = self.generator(z)
            
            # Loss measures generator's ability to fool the discriminator
            fake_validity = self.discriminator(fake_images)
            g_loss = -torch.mean(fake_validity)
            
            g_loss.backward()
            self.g_optimizer.step()
            
            return {
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
                "gradient_penalty": gradient_penalty.item()
            }
        
        return {
            "d_loss": d_loss.item(),
            "gradient_penalty": gradient_penalty.item()
        }
    
    def train(self, num_epochs: int, save_interval: int = 1000) -> None:
        """Train the GAN for the specified number of epochs."""
        self.n_critic_step = 0
        
        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()
            
            progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, real_images in enumerate(progress_bar):
                real_images = real_images.to(device)
                
                # Train step
                losses = self.train_step(real_images)
                
                # Update progress bar
                progress_bar.set_postfix(losses)
                
                # Log to tensorboard
                global_step = epoch * len(self.data_loader) + batch_idx
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f"loss/{loss_name}", loss_value, global_step)
                
                # Save checkpoints
                if global_step % save_interval == 0:
                    self.save_checkpoint(global_step)
                    
                    # Generate sample images
                    self.generate_samples(global_step)
                
                self.n_critic_step += 1
    
    def save_checkpoint(self, step: int) -> None:
        """Save model checkpoints."""
        checkpoint = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "step": step
        }
        
        torch.save(checkpoint, self.save_dir / f"checkpoint_{step}.pt")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        self.d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        
        return checkpoint["step"]
    
    def generate_samples(self, step: int, num_samples: int = 16) -> None:
        """Generate and save sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            fake_images = self.generator(z)
            
            # Denormalize images
            fake_images = (fake_images + 1) / 2
            
            # Create grid
            grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=False)
            
            # Save to tensorboard
            self.writer.add_image("generated_samples", grid, step)
            
            # Save to disk
            save_path = self.save_dir / f"samples_{step}.png"
            torchvision.utils.save_image(grid, save_path)
        
        self.generator.train()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the GAN on a test set."""
        self.generator.eval()
        self.discriminator.eval()
        
        real_scores = []
        fake_scores = []
        
        with torch.no_grad():
            for real_images in test_loader:
                real_images = real_images.to(device)
                batch_size = real_images.size(0)
                
                # Generate fake images
                z = torch.randn(batch_size, self.latent_dim).to(device)
                fake_images = self.generator(z)
                
                # Get discriminator scores
                real_validity = self.discriminator(real_images)
                fake_validity = self.discriminator(fake_images)
                
                real_scores.extend(real_validity.cpu().numpy())
                fake_scores.extend(fake_validity.cpu().numpy())
        
        real_scores = np.array(real_scores)
        fake_scores = np.array(fake_scores)
        
        # Calculate metrics
        real_mean = np.mean(real_scores)
        fake_mean = np.mean(fake_scores)
        
        # Calculate FID score (simplified)
        real_features = self._extract_features(real_images)
        fake_features = self._extract_features(fake_images)
        
        fid_score = self._calculate_fid(real_features, fake_features)
        
        return {
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "fid_score": fid_score
        }
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images using the discriminator."""
        features = []
        
        def hook(module, input, output):
            features.append(output)
        
        # Register hook to get intermediate features
        handle = self.discriminator.conv4.register_forward_hook(hook)
        
        with torch.no_grad():
            self.discriminator(images)
        
        handle.remove()
        
        return features[0].mean([2, 3])
    
    def _calculate_fid(self, real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
        """Calculate Fréchet Inception Distance (simplified version)."""
        real_mean = real_features.mean(0)
        fake_mean = fake_features.mean(0)
        
        real_cov = torch.cov(real_features.t())
        fake_cov = torch.cov(fake_features.t())
        
        diff = real_mean - fake_mean
        cov_mean = (real_cov + fake_cov) / 2
        
        fid = diff.dot(diff) + torch.trace(real_cov + fake_cov - 2 * torch.sqrt(real_cov @ fake_cov))
        
        return fid.item()
    
    def compare_models(self, other_trainer: 'GANTrainer', num_samples: int = 100) -> Dict[str, float]:
        """Compare this GAN with another GAN."""
        self.generator.eval()
        other_trainer.generator.eval()
        
        # Generate samples from both generators
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        with torch.no_grad():
            fake_images_self = self.generator(z)
            fake_images_other = other_trainer.generator(z)
        
        # Get discriminator scores
        with torch.no_grad():
            self_scores = self.discriminator(fake_images_self)
            other_scores = self.discriminator(fake_images_other)
            
            other_self_scores = other_trainer.discriminator(fake_images_self)
            other_other_scores = other_trainer.discriminator(fake_images_other)
        
        # Calculate metrics
        self_fool_rate = (self_scores < 0).float().mean().item()
        other_fool_rate = (other_scores < 0).float().mean().item()
        
        cross_detection_rate = (other_self_scores > 0).float().mean().item()
        other_cross_detection_rate = (other_other_scores > 0).float().mean().item()
        
        return {
            f"{self.quality}_generator_fools_{self.quality}_discriminator": self_fool_rate,
            f"{other_trainer.quality}_generator_fools_{self.quality}_discriminator": other_fool_rate,
            f"{self.quality}_generator_fools_{other_trainer.quality}_discriminator": cross_detection_rate,
            f"{other_trainer.quality}_generator_fools_{other_trainer.quality}_discriminator": other_cross_detection_rate
        } 