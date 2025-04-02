import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from tqdm import tqdm
import random

class VideoFrameDataset(Dataset):
    """Dataset for video frames."""
    
    def __init__(
        self,
        video_dir: Union[str, Path],
        frame_size: Tuple[int, int] = (256, 256),
        frame_count: int = 16,
        transform: Optional[transforms.Compose] = None,
        cache_frames: bool = True
    ):
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        self.frame_count = frame_count
        self.cache_frames = cache_frames
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        
        # Find all video files
        self.video_files = self._find_video_files()
        
        # Extract frames from videos
        self.frames, self.frame_paths = self._extract_frames()
        
        print(f"Loaded {len(self.frames)} frames from {len(self.video_files)} videos")
    
    def _find_video_files(self) -> List[Path]:
        """Find all video files in the directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(self.video_dir.glob(f"**/*{ext}")))
        
        return video_files
    
    def _extract_frames(self) -> Tuple[List[np.ndarray], List[Path]]:
        """Extract frames from videos."""
        frames = []
        frame_paths = []
        
        for video_path in tqdm(self.video_files, desc="Extracting frames"):
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"Error opening video: {video_path}")
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame indices to extract
            if frame_count <= self.frame_count:
                frame_indices = list(range(frame_count))
            else:
                # Extract frames evenly spaced throughout the video
                frame_indices = np.linspace(0, frame_count - 1, self.frame_count, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame = cv2.resize(frame, self.frame_size)
                
                frames.append(frame)
                frame_paths.append(video_path)
            
            cap.release()
        
        return frames, frame_paths
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        
        # Apply transform
        frame_tensor = self.transform(frame)
        
        return frame_tensor
    
    def get_video_frames(self, video_path: Path) -> List[torch.Tensor]:
        """Get all frames from a specific video."""
        frames = []
        
        for i, path in enumerate(self.frame_paths):
            if path == video_path:
                frames.append(self[i])
        
        return frames

class VideoDataset(Dataset):
    """Dataset for loading video clips with temporal consistency."""
    def __init__(
        self,
        video_dir: str,
        frame_size: Tuple[int, int],
        frame_count: int = 16,
        transform=None
    ):
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        self.frame_count = frame_count
        self.transform = transform
        
        # Get all video files (including .webm)
        self.video_files = list(self.video_dir.glob("*.mp4")) + \
                          list(self.video_dir.glob("*.avi")) + \
                          list(self.video_dir.glob("*.mov")) + \
                          list(self.video_dir.glob("*.webm"))
        
        if not self.video_files:
            raise ValueError(f"No video files found in {video_dir}")
        
        print(f"Found {len(self.video_files)} videos in {video_dir}")
    
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame indices for a random clip
        max_start = max(0, total_frames - self.frame_count)
        start_frame = random.randint(0, max_start)
        
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(self.frame_count):
            ret, frame = cap.read()
            if not ret:
                # If we reach the end of the video, loop back to the start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame to target size
            frame = cv2.resize(frame, self.frame_size)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
        
        cap.release()
        
        # Stack frames and convert to tensor
        frames = np.stack(frames)  # [T, H, W, C]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # [C, T, H, W]
        
        if self.transform:
            frames = self.transform(frames)
        
        return frames

def create_data_loaders(
    high_quality_dir: str,
    low_quality_dir: str,
    frame_size: Tuple[int, int],
    frame_count: int = 16,
    batch_size: int = 4,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for high and low quality video data.
    
    Args:
        high_quality_dir: Directory containing high quality videos
        low_quality_dir: Directory containing low quality videos
        frame_size: Size of frames (height, width)
        frame_count: Number of frames to extract from each video
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (high_quality_loader, low_quality_loader)
    """
    # Create datasets
    high_quality_dataset = VideoDataset(
        video_dir=high_quality_dir,
        frame_size=frame_size,
        frame_count=frame_count
    )
    
    low_quality_dataset = VideoDataset(
        video_dir=low_quality_dir,
        frame_size=frame_size,
        frame_count=frame_count
    )
    
    # Create data loaders
    high_quality_loader = DataLoader(
        high_quality_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    low_quality_loader = DataLoader(
        low_quality_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return high_quality_loader, low_quality_loader 