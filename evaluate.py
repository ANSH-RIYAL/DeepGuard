import torch
import torch.nn as nn
import argparse
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.base_gan import Generator, Discriminator
from utils.training import GANTrainer
from utils.dataset import create_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GAN models for deepfake detection")
    
    # Data arguments
    parser.add_argument("--high_quality_dir", type=str, default="data/high_quality_data", 
                        help="Directory containing high quality video data")
    parser.add_argument("--low_quality_dir", type=str, default="data/low_quality_data", 
                        help="Directory containing low quality video data")
    parser.add_argument("--frame_size", type=int, nargs=2, default=[256, 256], 
                        help="Frame size (height, width)")
    parser.add_argument("--frame_count", type=int, default=16, 
                        help="Number of frames to extract from each video")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for evaluation")
    
    # Model arguments
    parser.add_argument("--latent_dim", type=int, default=512, 
                        help="Dimension of latent space")
    parser.add_argument("--style_dim", type=int, default=512, 
                        help="Dimension of style space")
    parser.add_argument("--channels", type=int, default=3, 
                        help="Number of channels in images")
    
    # Checkpoint arguments
    parser.add_argument("--high_quality_checkpoint", type=str, required=True, 
                        help="Path to high quality GAN checkpoint")
    parser.add_argument("--low_quality_checkpoint", type=str, required=True, 
                        help="Path to low quality GAN checkpoint")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Directory for output files")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Number of samples to generate for evaluation")
    
    return parser.parse_args()

def generate_samples(generator, num_samples, latent_dim, output_dir, prefix):
    """Generate and save samples from the generator."""
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(generator.device)
        fake_images = generator(z)
        
        # Denormalize images
        fake_images = (fake_images + 1) / 2
        
        # Save samples
        for i in range(num_samples):
            plt.figure(figsize=(10, 10))
            plt.imshow(fake_images[i].permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f"{prefix}_sample_{i}.png"))
            plt.close()

def evaluate_discriminator(discriminator, data_loader, output_dir, prefix):
    """Evaluate the discriminator on real and fake images."""
    discriminator.eval()
    
    real_scores = []
    fake_scores = []
    
    with torch.no_grad():
        for real_images in tqdm(data_loader, desc=f"Evaluating {prefix} discriminator"):
            real_images = real_images.to(discriminator.device)
            batch_size = real_images.size(0)
            
            # Get discriminator scores for real images
            real_validity = discriminator(real_images)
            real_scores.extend(real_validity.cpu().numpy())
            
            # Generate fake images
            z = torch.randn(batch_size, discriminator.latent_dim).to(discriminator.device)
            fake_images = discriminator.generator(z)
            
            # Get discriminator scores for fake images
            fake_validity = discriminator(fake_images)
            fake_scores.extend(fake_validity.cpu().numpy())
    
    # Calculate metrics
    real_mean = np.mean(real_scores)
    fake_mean = np.mean(fake_scores)
    
    # Plot score distributions
    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=50, alpha=0.5, label="Real")
    plt.hist(fake_scores, bins=50, alpha=0.5, label="Fake")
    plt.xlabel("Discriminator Score")
    plt.ylabel("Count")
    plt.title(f"{prefix} Discriminator Score Distribution")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{prefix}_score_distribution.png"))
    plt.close()
    
    return {
        "real_mean": real_mean,
        "fake_mean": fake_mean,
        "real_scores": real_scores,
        "fake_scores": fake_scores
    }

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loaders
    high_quality_loader, low_quality_loader = create_data_loaders(
        high_quality_dir=args.high_quality_dir,
        low_quality_dir=args.low_quality_dir,
        batch_size=args.batch_size,
        frame_size=tuple(args.frame_size),
        frame_count=args.frame_count
    )
    
    # Create models
    high_quality_generator = Generator(
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        channels=args.channels,
        max_resolution=max(args.frame_size)
    )
    
    high_quality_discriminator = Discriminator(
        channels=args.channels,
        max_resolution=max(args.frame_size)
    )
    
    low_quality_generator = Generator(
        latent_dim=args.latent_dim,
        style_dim=args.style_dim,
        channels=args.channels,
        max_resolution=max(args.frame_size)
    )
    
    low_quality_discriminator = Discriminator(
        channels=args.channels,
        max_resolution=max(args.frame_size)
    )
    
    # Load checkpoints
    high_quality_checkpoint = torch.load(args.high_quality_checkpoint)
    low_quality_checkpoint = torch.load(args.low_quality_checkpoint)
    
    high_quality_generator.load_state_dict(high_quality_checkpoint["generator"])
    high_quality_discriminator.load_state_dict(high_quality_checkpoint["discriminator"])
    
    low_quality_generator.load_state_dict(low_quality_checkpoint["generator"])
    low_quality_discriminator.load_state_dict(low_quality_checkpoint["discriminator"])
    
    # Create trainers
    high_quality_trainer = GANTrainer(
        generator=high_quality_generator,
        discriminator=high_quality_discriminator,
        data_loader=high_quality_loader,
        latent_dim=args.latent_dim,
        quality="high"
    )
    
    low_quality_trainer = GANTrainer(
        generator=low_quality_generator,
        discriminator=low_quality_discriminator,
        data_loader=low_quality_loader,
        latent_dim=args.latent_dim,
        quality="low"
    )
    
    # Generate samples
    print("Generating high quality samples...")
    generate_samples(
        high_quality_generator,
        args.num_samples,
        args.latent_dim,
        args.output_dir,
        "high_quality"
    )
    
    print("Generating low quality samples...")
    generate_samples(
        low_quality_generator,
        args.num_samples,
        args.latent_dim,
        args.output_dir,
        "low_quality"
    )
    
    # Evaluate discriminators
    print("Evaluating high quality discriminator...")
    high_quality_results = evaluate_discriminator(
        high_quality_discriminator,
        high_quality_loader,
        args.output_dir,
        "high_quality"
    )
    
    print("Evaluating low quality discriminator...")
    low_quality_results = evaluate_discriminator(
        low_quality_discriminator,
        low_quality_loader,
        args.output_dir,
        "low_quality"
    )
    
    # Compare models
    print("Comparing models...")
    comparison_results = high_quality_trainer.compare_models(low_quality_trainer)
    
    # Save results
    results = {
        "high_quality": high_quality_results,
        "low_quality": low_quality_results,
        "comparison": comparison_results
    }
    
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print("Evaluation results:")
    print(f"High quality discriminator - Real mean: {high_quality_results['real_mean']:.4f}, Fake mean: {high_quality_results['fake_mean']:.4f}")
    print(f"Low quality discriminator - Real mean: {low_quality_results['real_mean']:.4f}, Fake mean: {low_quality_results['fake_mean']:.4f}")
    print("\nComparison results:")
    for key, value in comparison_results.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main() 