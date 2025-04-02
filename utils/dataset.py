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
    """Dataset for video clips."""
    
    def __init__(
        self,
        video_dir: Union[str, Path],
        frame_size: Tuple[int, int] = (256, 256),
        clip_length: int = 16,
        transform: Optional[transforms.Compose] = None
    ):
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        self.clip_length = clip_length
        
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
        
        # Extract clips from videos
        self.clips = self._extract_clips()
        
        print(f"Loaded {len(self.clips)} clips from {len(self.video_files)} videos")
    
    def _find_video_files(self) -> List[Path]:
        """Find all video files in the directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(self.video_dir.glob(f"**/*{ext}")))
        
        return video_files
    
    def _extract_clips(self) -> List[List[np.ndarray]]:
        """Extract clips from videos."""
        clips = []
        
        for video_path in tqdm(self.video_files, desc="Extracting clips"):
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                print(f"Error opening video: {video_path}")
                continue
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract clips
            for start_frame in range(0, frame_count - self.clip_length, self.clip_length // 2):
                clip_frames = []
                
                for i in range(self.clip_length):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize frame
                    frame = cv2.resize(frame, self.frame_size)
                    
                    clip_frames.append(frame)
                
                if len(clip_frames) == self.clip_length:
                    clips.append(clip_frames)
            
            cap.release()
        
        return clips
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        clip = self.clips[idx]
        
        # Apply transform to each frame
        clip_tensors = [self.transform(frame) for frame in clip]
        
        # Stack frames into a single tensor
        clip_tensor = torch.stack(clip_tensors)
        
        return clip_tensor

def create_data_loaders(
    high_quality_dir: Union[str, Path],
    low_quality_dir: Union[str, Path],
    batch_size: int = 8,
    frame_size: Tuple[int, int] = (256, 256),
    frame_count: int = 16,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for high and low quality data."""
    # Create datasets
    high_quality_dataset = VideoFrameDataset(
        video_dir=high_quality_dir,
        frame_size=frame_size,
        frame_count=frame_count
    )
    
    low_quality_dataset = VideoFrameDataset(
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