import os
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run GAN training and evaluation experiment")
    
    # Data arguments
    parser.add_argument("--high_quality_dir", type=str, default="data/high_quality_data", 
                        help="Directory containing high quality video data")
    parser.add_argument("--low_quality_dir", type=str, default="data/low_quality_data", 
                        help="Directory containing low quality video data")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="experiment_results", 
                        help="Directory for output files")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    print("Starting training...")
    train_cmd = [
        "python", "train.py",
        "--high_quality_dir", args.high_quality_dir,
        "--low_quality_dir", args.low_quality_dir,
        "--num_epochs", str(args.num_epochs),
        "--batch_size", str(args.batch_size),
        "--output_dir", os.path.join(args.output_dir, "training")
    ]
    
    subprocess.run(train_cmd)
    
    # Find checkpoints
    training_dir = Path(args.output_dir) / "training"
    high_quality_checkpoint = list(training_dir.glob("high/checkpoint_*.pt"))[-1]
    low_quality_checkpoint = list(training_dir.glob("low/checkpoint_*.pt"))[-1]
    
    # Run evaluation
    print("Starting evaluation...")
    eval_cmd = [
        "python", "evaluate.py",
        "--high_quality_dir", args.high_quality_dir,
        "--low_quality_dir", args.low_quality_dir,
        "--high_quality_checkpoint", str(high_quality_checkpoint),
        "--low_quality_checkpoint", str(low_quality_checkpoint),
        "--output_dir", os.path.join(args.output_dir, "evaluation")
    ]
    
    subprocess.run(eval_cmd)
    
    print(f"Experiment completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 