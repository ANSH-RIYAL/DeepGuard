import torch
import torch.nn as nn
import argparse
from pathlib import Path
import os
import json
from tqdm import tqdm

from models.base_gan import Generator, Discriminator
from utils.training import GANTrainer
from utils.dataset import create_data_loaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN models for deepfake detection")
    
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
                        help="Batch size for training")
    
    # Model arguments
    parser.add_argument("--latent_dim", type=int, default=512, 
                        help="Dimension of latent space")
    parser.add_argument("--style_dim", type=int, default=512, 
                        help="Dimension of style space")
    parser.add_argument("--channels", type=int, default=3, 
                        help="Number of channels in images")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument("--lr_g", type=float, default=0.0002, 
                        help="Learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=0.0002, 
                        help="Learning rate for discriminator")
    parser.add_argument("--beta1", type=float, default=0.5, 
                        help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, 
                        help="Beta2 for Adam optimizer")
    parser.add_argument("--lambda_gp", type=float, default=10.0, 
                        help="Gradient penalty lambda")
    parser.add_argument("--n_critic", type=int, default=5, 
                        help="Number of D updates per G update")
    parser.add_argument("--save_interval", type=int, default=1000, 
                        help="Interval for saving checkpoints")
    
    # Output arguments
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="Directory for tensorboard logs")
    parser.add_argument("--save_dir", type=str, default="checkpoints", 
                        help="Directory for saving checkpoints")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory for output files")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
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
    
    # Create trainers
    high_quality_trainer = GANTrainer(
        generator=high_quality_generator,
        discriminator=high_quality_discriminator,
        data_loader=high_quality_loader,
        latent_dim=args.latent_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda_gp=args.lambda_gp,
        n_critic=args.n_critic,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        quality="high"
    )
    
    low_quality_trainer = GANTrainer(
        generator=low_quality_generator,
        discriminator=low_quality_discriminator,
        data_loader=low_quality_loader,
        latent_dim=args.latent_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda_gp=args.lambda_gp,
        n_critic=args.n_critic,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        quality="low"
    )
    
    # Train models
    print("Training high quality GAN...")
    high_quality_trainer.train(
        num_epochs=args.num_epochs,
        save_interval=args.save_interval
    )
    
    print("Training low quality GAN...")
    low_quality_trainer.train(
        num_epochs=args.num_epochs,
        save_interval=args.save_interval
    )
    
    # Compare models
    print("Comparing models...")
    comparison_results = high_quality_trainer.compare_models(low_quality_trainer)
    
    # Save comparison results
    with open(os.path.join(args.output_dir, "comparison_results.json"), "w") as f:
        json.dump(comparison_results, f, indent=4)
    
    print("Comparison results:")
    for key, value in comparison_results.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main() 