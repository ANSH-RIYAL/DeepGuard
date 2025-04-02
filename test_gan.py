import torch
import torch.nn as nn
from models.base_gan import Generator, Discriminator

def test_gan_forward_pass():
    print("Testing GAN forward pass...")
    
    # Model parameters
    batch_size = 2
    latent_dim = 512
    num_channels = 3
    num_frames = 16
    frame_size = (256, 256)
    
    # Initialize models
    generator = Generator(
        latent_dim=latent_dim,
        num_channels=num_channels,
        frame_size=frame_size,
        num_frames=num_frames
    )
    
    discriminator = Discriminator(
        num_channels=num_channels,
        frame_size=frame_size,
        num_frames=num_frames
    )
    
    # Generate random latent vectors
    z = torch.randn(batch_size, latent_dim)
    print(f"Input latent shape: {z.shape}")
    
    # Generate fake videos
    fake_videos = generator(z)
    print(f"Generated videos shape: {fake_videos.shape}")
    
    # Get discriminator output
    disc_output = discriminator(fake_videos)
    print(f"Discriminator output shape: {disc_output.shape}")
    
    # Test with dummy "real" videos
    real_videos = torch.randn(batch_size, num_channels, num_frames, *frame_size)  # [B, C, T, H, W]
    print(f"\nReal videos shape: {real_videos.shape}")
    
    # Reshape real videos to match discriminator's expected input
    real_videos = real_videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
    real_videos = real_videos.reshape(batch_size, -1, frame_size[0], frame_size[1])  # [B, T*C, H, W]
    print(f"Reshaped real videos shape: {real_videos.shape}")
    
    # Get discriminator output for real videos
    disc_output_real = discriminator(real_videos)
    print(f"Discriminator output shape (real): {disc_output_real.shape}")
    
    print("\nAll shapes look correct!" if all(x.numel() > 0 for x in [fake_videos, disc_output, disc_output_real]) else "Something went wrong!")

if __name__ == "__main__":
    test_gan_forward_pass() 