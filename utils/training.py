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
import cv2

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
        self.generator = generator
        self.discriminator = discriminator
        self.data_loader = data_loader
        self.latent_dim = latent_dim
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.quality = quality
        
        # Setup optimizers
        self.optimizer_g = optim.Adam(
            generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        
        # Setup tensorboard
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(os.path.join(log_dir, quality))
        
        # Setup checkpoint directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Move models to device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # Loss functions
        self.criterion = nn.BCEWithLogitsLoss()
        
    def compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1).to(self.device)
        
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_step(self, real_videos: torch.Tensor) -> dict:
        """Perform one training step."""
        batch_size = real_videos.size(0)
        
        # Reshape real videos to match discriminator's expected input
        real_videos = real_videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        real_videos = real_videos.reshape(batch_size, -1, real_videos.shape[-2], real_videos.shape[-1])  # [B, T*C, H, W]
        
        # Train discriminator
        for _ in range(self.n_critic):
            self.optimizer_d.zero_grad()
            
            # Generate fake videos
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_videos = self.generator(z)
            
            # Compute discriminator loss
            real_validity = self.discriminator(real_videos)
            fake_validity = self.discriminator(fake_videos.detach())
            
            gradient_penalty = self.compute_gradient_penalty(real_videos, fake_videos.detach())
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
            
            d_loss.backward()
            self.optimizer_d.step()
        
        # Train generator
        self.optimizer_g.zero_grad()
        
        # Generate fake videos
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_videos = self.generator(z)
        
        # Compute generator loss
        fake_validity = self.discriminator(fake_videos)
        g_loss = -torch.mean(fake_validity)
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "gradient_penalty": gradient_penalty.item()
        }
    
    def train(self, num_epochs: int, save_interval: int = 1000):
        """Train the GAN."""
        for epoch in range(num_epochs):
            for batch_idx, real_videos in enumerate(self.data_loader):
                real_videos = real_videos.to(self.device)
                
                # Train step
                losses = self.train_step(real_videos)
                
                # Log losses
                step = epoch * len(self.data_loader) + batch_idx
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f"losses/{loss_name}", loss_value, step)
                
                # Save samples
                if batch_idx % save_interval == 0:
                    self.save_samples(step)
                
                # Save checkpoint
                if batch_idx % save_interval == 0:
                    self.save_checkpoint(step)
                
                # Print progress
                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(self.data_loader)}] "
                          f"D_loss: {losses['d_loss']:.4f} G_loss: {losses['g_loss']:.4f}")
    
    def save_samples(self, step: int, num_samples: int = 4):
        """Save generated video samples."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_videos = self.generator(z)
            
            # Convert to numpy and save
            fake_videos = fake_videos.cpu().numpy()
            
            # Create output directory
            samples_dir = Path("samples") / self.quality
            samples_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each video
            for i in range(num_samples):
                video_path = samples_dir / f"sample_{step}_{i}.mp4"
                
                # Convert to uint8
                video = (fake_videos[i] * 255).astype(np.uint8)
                
                # Save video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (video.shape[3], video.shape[2]))
                
                for frame in video.transpose(2, 3, 1, 0):  # (H, W, C, T) -> (H, W, C)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)
                
                out.release()
        
        self.generator.train()
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "step": step
        }
        
        checkpoint_path = self.save_dir / f"{self.quality}_checkpoint_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        
        return checkpoint["step"]
    
    def compare_models(self, other_trainer: 'GANTrainer') -> dict:
        """Compare this model with another model."""
        self.generator.eval()
        other_trainer.generator.eval()
        
        with torch.no_grad():
            # Generate samples from both models
            z = torch.randn(4, self.latent_dim).to(self.device)
            
            fake_videos1 = self.generator(z)
            fake_videos2 = other_trainer.generator(z)
            
            # Compute discriminator scores
            scores1 = self.discriminator(fake_videos1)
            scores2 = other_trainer.discriminator(fake_videos1)
            
            scores3 = self.discriminator(fake_videos2)
            scores4 = other_trainer.discriminator(fake_videos2)
        
        self.generator.train()
        other_trainer.generator.train()
        
        return {
            "high_quality_fooling_rate": (scores2 > 0).float().mean().item(),
            "low_quality_fooling_rate": (scores3 > 0).float().mean().item(),
            "high_quality_discriminator_score": scores1.mean().item(),
            "low_quality_discriminator_score": scores4.mean().item()
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the GAN on a test set."""
        self.generator.eval()
        self.discriminator.eval()
        
        real_scores = []
        fake_scores = []
        
        with torch.no_grad():
            for real_videos in test_loader:
                real_videos = real_videos.to(device)
                batch_size = real_videos.size(0)
                
                # Generate fake videos
                z = torch.randn(batch_size, self.latent_dim).to(device)
                fake_videos = self.generator(z)
                
                # Get discriminator scores
                real_validity = self.discriminator(real_videos)
                fake_validity = self.discriminator(fake_videos)
                
                real_scores.extend(real_validity.cpu().numpy())
                fake_scores.extend(fake_validity.cpu().numpy())
        
        real_scores = np.array(real_scores)
        fake_scores = np.array(fake_scores)
        
        # Calculate metrics
        real_mean = np.mean(real_scores)
        fake_mean = np.mean(fake_scores)
        
        # Calculate FID score (simplified)
        real_features = self._extract_features(real_videos)
        fake_features = self._extract_features(fake_videos)
        
        fid_score = self._calculate_fid(real_features, fake_features)
        
        return {
            "real_mean": real_mean,
            "fake_mean": fake_mean,
            "fid_score": fid_score
        }
    
    def _extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract features from videos using the discriminator."""
        features = []
        
        def hook(module, input, output):
            features.append(output)
        
        # Register hook to get intermediate features
        handle = self.discriminator.conv4.register_forward_hook(hook)
        
        with torch.no_grad():
            self.discriminator(videos)
        
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
    
    def generate_samples(self, step: int, num_samples: int = 16) -> None:
        """Generate and save sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            fake_videos = self.generator(z)
            
            # Denormalize videos
            fake_videos = (fake_videos + 1) / 2
            
            # Create grid
            grid = torchvision.utils.make_grid(fake_videos, nrow=4, normalize=False)
            
            # Save to tensorboard
            self.writer.add_image("generated_samples", grid, step)
            
            # Save to disk
            save_path = self.save_dir / f"samples_{step}.png"
            torchvision.utils.save_image(grid, save_path)
        
        self.generator.train()
    
    def compare_models(self, other_trainer: 'GANTrainer', num_samples: int = 100) -> Dict[str, float]:
        """Compare this GAN with another GAN."""
        self.generator.eval()
        other_trainer.generator.eval()
        
        # Generate samples from both generators
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        with torch.no_grad():
            fake_videos_self = self.generator(z)
            fake_videos_other = other_trainer.generator(z)
        
        # Get discriminator scores
        with torch.no_grad():
            self_scores = self.discriminator(fake_videos_self)
            other_scores = self.discriminator(fake_videos_other)
            
            other_self_scores = other_trainer.discriminator(fake_videos_self)
            other_other_scores = other_trainer.discriminator(fake_videos_other)
        
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