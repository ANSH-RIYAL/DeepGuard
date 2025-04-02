# DeepGuard: GAN-based Deepfake Detection

DeepGuard is a deep learning-based system for detecting deepfake videos using Generative Adversarial Networks (GANs). The system trains separate GANs on high-quality and low-quality video data to learn the distinguishing features of each quality level.

## Project Hypothesis

The core hypothesis of this project is:
> Across all machine learning and deep learning architectures for deepfake generation, using the same architecture with a high-quality dataset produces a better generator and detector combination (superior to the generator and detector created using a low-quality dataset).

This hypothesis will be tested by:
1. Training identical architectures on different quality datasets
2. Comparing the performance metrics between high and low-quality models
3. Evaluating the models on various deepfake datasets

### Planned Dataset Comparisons

The system will be evaluated against:
1. FaceForensics++ (high-quality deepfake dataset)
2. DFDC (Facebook's DeepFake Detection Challenge dataset)
3. Celeb-DF (Celebrity DeepFake dataset)
4. UADFV (University of Albany DeepFake Video dataset)

## Features

- Dual GAN architecture for high and low-quality video analysis
- Temporal consistency in video generation
- Cross-evaluation between quality levels
- Comprehensive model comparison metrics
- Support for various video formats and resolutions
- TensorBoard integration for training visualization

## Project Structure

```
DeepGuard/
├── data/                  # Video data directory (not tracked by git)
│   ├── high_quality_data/ # High-quality video samples
│   └── low_quality_data/  # Low-quality video samples
├── models/               # Model architecture definitions
│   └── base_gan.py      # Base GAN implementation
├── utils/               # Utility functions
│   ├── dataset.py       # Video dataset handling
│   ├── training.py      # Training utilities
│   └── pretrained.py    # Pre-trained model utilities
├── samples/             # Generated video samples (not tracked by git)
├── checkpoints/         # Model checkpoints (not tracked by git)
├── logs/               # Training logs (not tracked by git)
├── output/             # Evaluation results (not tracked by git)
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── test_gan.py         # GAN testing script
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ANSH-RIYAL/DeepGuard.git
cd DeepGuard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the GANs on your video data:

```bash
python train.py --high_quality_dir data/high_quality_data \
                --low_quality_dir data/low_quality_data \
                --frame_size 256 256 \
                --batch_size 4 \
                --num_epochs 50
```

### Evaluation

To evaluate the trained models:

```bash
python evaluate.py --high_quality_checkpoint checkpoints/high_quality_latest.pt \
                  --low_quality_checkpoint checkpoints/low_quality_latest.pt \
                  --num_samples 100
```

### Testing

To test the GAN architecture:

```bash
python test_gan.py
```

## Model Architecture

The system uses a dual GAN architecture:

1. **Generator**:
   - Input: Random latent vector (512 dimensions)
   - Output: Video frames (16 frames, 256x256 resolution)
   - Architecture: Style-based generator with temporal consistency

2. **Discriminator**:
   - Input: Video frames
   - Output: Real/fake probability
   - Architecture: Convolutional network with temporal awareness

## Training Process

1. The system trains two separate GANs:
   - High-quality GAN on high-quality video data
   - Low-quality GAN on low-quality video data

2. Training metrics:
   - Generator loss
   - Discriminator loss
   - Gradient penalty
   - Cross-evaluation scores

3. Model comparison:
   - Cross-discriminator evaluation
   - FID score calculation
   - Quality metrics

## Results

The system generates comparison metrics between high and low-quality models, including:
- Discriminator scores on both quality levels
- Cross-evaluation results
- FID scores
- Generated sample quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is proprietary software. All rights reserved. No part of this software may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the copyright holder.

## Acknowledgments

- NVIDIA StyleGAN3 for architecture inspiration
- OpenCV for video processing
- PyTorch for deep learning framework 