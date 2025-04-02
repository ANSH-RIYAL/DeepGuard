import os
import torch
import requests
from pathlib import Path
from tqdm import tqdm
from models.base_gan import Generator, Discriminator

STYLEGAN3_URLS = {
    'ffhq-256': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-256x256.pkl',
    'ffhq-1024': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl',
    'afhq-512': 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhq-512x512.pkl'
}

def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def load_pretrained_weights(
    model_type: str = 'ffhq-256',
    cache_dir: str = 'pretrained_weights'
) -> tuple[Generator, Discriminator]:
    """
    Load pre-trained StyleGAN3 weights and convert them to our model format.
    
    Args:
        model_type: Type of pre-trained model to load ('ffhq-256', 'ffhq-1024', 'afhq-512')
        cache_dir: Directory to store downloaded weights
    
    Returns:
        Tuple of (Generator, Discriminator) with pre-trained weights
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Download weights if not already present
    weights_path = cache_dir / f"{model_type}.pkl"
    if not weights_path.exists():
        print(f"Downloading {model_type} weights...")
        download_file(STYLEGAN3_URLS[model_type], weights_path)
    
    # Load the weights
    print(f"Loading {model_type} weights...")
    with open(weights_path, 'rb') as f:
        pretrained = torch.load(f, map_location='cpu')
    
    # Create our models
    generator = Generator(
        latent_dim=512,
        style_dim=512,
        channels=3,
        max_resolution=256 if '256' in model_type else 512 if '512' in model_type else 1024
    )
    
    discriminator = Discriminator(
        channels=3,
        max_resolution=256 if '256' in model_type else 512 if '512' in model_type else 1024
    )
    
    # Convert and load weights
    # Note: This is a simplified conversion. You might need to adjust the mapping
    # based on the exact architecture differences between StyleGAN3 and our implementation
    generator_state = {}
    discriminator_state = {}
    
    for k, v in pretrained['G'].items():
        if 'mapping' in k:
            generator_state[k.replace('mapping.', '')] = v
        elif 'synthesis' in k:
            generator_state[k.replace('synthesis.', '')] = v
    
    for k, v in pretrained['D'].items():
        discriminator_state[k] = v
    
    generator.load_state_dict(generator_state, strict=False)
    discriminator.load_state_dict(discriminator_state, strict=False)
    
    print("Successfully loaded pre-trained weights")
    return generator, discriminator 