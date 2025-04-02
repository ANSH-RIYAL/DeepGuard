import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

from models.base_gan import Generator, Discriminator
from utils.dataset import create_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN for deepfake detection')
    parser.add_argument('--high_quality_dir', type=str, default='data/high_quality_data',
                        help='Directory containing high-quality video data')
    parser.add_argument('--low_quality_dir', type=str, default='data/low_quality_data',
                        help='Directory containing low-quality video data')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[256, 256],
                        help='Frame size (height, width)')
    parser.add_argument('--frame_count', type=int, default=16,
                        help='Number of frames per video clip')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr_g', type=float, default=0.0002,
                        help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=0.0002,
                        help='Learning rate for discriminator')
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Dimension of latent space')
    parser.add_argument('--style_dim', type=int, default=512,
                        help='Dimension of style space')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Gradient penalty coefficient')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory for model checkpoints')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory for output files')
    return parser.parse_args()

class GANTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        quality: str,
        latent_dim: int,
        lr_g: float,
        lr_d: float,
        n_critic: int,
        lambda_gp: float,
        log_dir: str,
        save_dir: str
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.data_loader = data_loader
        self.device = device
        self.quality = quality
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        
        # Optimizers
        self.optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.99))
        self.optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.99))
        
        # Tensorboard
        self.writer = SummaryWriter(os.path.join(log_dir, quality))
        
        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_gradient_penalty(self, real_videos: torch.Tensor, fake_videos: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_videos.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        
        # Interpolate between real and fake videos
        interpolates = (alpha * real_videos + (1 - alpha) * fake_videos).requires_grad_(True)
        
        # Compute discriminator output
        d_interpolates = self.discriminator(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
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
            for batch_idx, real_videos in enumerate(tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
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
                    print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(self.data_loader)}] "
                          f"D_loss: {losses['d_loss']:.4f} G_loss: {losses['g_loss']:.4f}")
    
    def save_samples(self, step: int, num_samples: int = 4):
        """Save generated video samples."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            fake_videos = self.generator(z)
            
            # Print shape for debugging
            print(f"Generated video shape: {fake_videos.shape}")
            
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
                print(f"Video shape after selecting sample {i}: {video.shape}")
                
                # Reshape video from (C*T, H, W) to (T, H, W, C)
                num_frames = 16  # We know this from the frame_count parameter
                channels = 3
                height, width = video.shape[1], video.shape[2]
                video = video.reshape(num_frames, channels, height, width)
                video = video.transpose(0, 2, 3, 1)  # (T, H, W, C)
                
                # Save video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
                
                for frame in video:
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
        self.discriminator.eval()
        other_trainer.generator.eval()
        other_trainer.discriminator.eval()
        
        with torch.no_grad():
            # Generate samples from both models
            z = torch.randn(100, self.latent_dim).to(self.device)
            samples_this = self.generator(z)
            samples_other = other_trainer.generator(z)
            
            # Evaluate samples with both discriminators
            d_this_on_this = self.discriminator(samples_this).mean().item()
            d_this_on_other = self.discriminator(samples_other).mean().item()
            d_other_on_this = other_trainer.discriminator(samples_this).mean().item()
            d_other_on_other = other_trainer.discriminator(samples_other).mean().item()
            
            # Compute FID score (placeholder - we'll need to implement this properly)
            fid_score = 0.0  # TODO: Implement FID score calculation
            
            comparison = {
                f"{self.quality}_discriminator_on_{self.quality}": d_this_on_this,
                f"{self.quality}_discriminator_on_{other_trainer.quality}": d_this_on_other,
                f"{other_trainer.quality}_discriminator_on_{self.quality}": d_other_on_this,
                f"{other_trainer.quality}_discriminator_on_{other_trainer.quality}": d_other_on_other,
                "fid_score": fid_score
            }
            
            # Save comparison results
            comparison_path = self.save_dir / "model_comparison.json"
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=4)
            
            return comparison
        
        self.generator.train()
        self.discriminator.train()
        other_trainer.generator.train()
        other_trainer.discriminator.train()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    high_quality_loader, low_quality_loader = create_data_loaders(
        args.high_quality_dir,
        args.low_quality_dir,
        args.frame_size,
        args.frame_count,
        args.batch_size
    )
    
    # Initialize models
    high_quality_generator = Generator(
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        num_channels=3,
        frame_size=tuple(args.frame_size),
        num_frames=args.frame_count
    )
    
    high_quality_discriminator = Discriminator(
        num_channels=3,
        frame_size=tuple(args.frame_size),
        num_frames=args.frame_count
    )
    
    low_quality_generator = Generator(
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        num_channels=3,
        frame_size=tuple(args.frame_size),
        num_frames=args.frame_count
    )
    
    low_quality_discriminator = Discriminator(
        num_channels=3,
        frame_size=tuple(args.frame_size),
        num_frames=args.frame_count
    )
    
    # Initialize trainers
    high_quality_trainer = GANTrainer(
        generator=high_quality_generator,
        discriminator=high_quality_discriminator,
        data_loader=high_quality_loader,
        device=device,
        quality='high_quality',
        latent_dim=args.latent_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )
    
    low_quality_trainer = GANTrainer(
        generator=low_quality_generator,
        discriminator=low_quality_discriminator,
        data_loader=low_quality_loader,
        device=device,
        quality='low_quality',
        latent_dim=args.latent_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )
    
    # Train models
    print("Training high quality GAN...")
    high_quality_trainer.train(args.num_epochs)
    
    print("Training low quality GAN...")
    low_quality_trainer.train(args.num_epochs)
    
    # Compare models
    print("Comparing models...")
    comparison = high_quality_trainer.compare_models(low_quality_trainer)
    
    # Save comparison results
    with open(os.path.join(args.output_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print("Training complete!")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 