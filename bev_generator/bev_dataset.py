#!/usr/bin/env python3
"""
BEV Dataset for loading LiDAR, camera images, and ground truth BEV from V2Xverse dataset.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from utils.lidar_topview import lidar_to_topview


class BEVDataset(Dataset):
    """
    Dataset that returns:
    - LiDAR top-view image (generated from point cloud)
    - Camera images (4 views)
    - Ground truth BEV image (from birdview folder)
    
    Expected directory structure:
    ├── data_root/
    │   ├── weather-0/data/
    │   │   ├── routes_town05_xxx/
    │   │   │   ├── ego_vehicle_0/
    │   │   │   │   ├── lidar/
    │   │   │   │   │   ├── 0000.npy
    │   │   │   │   │   └── ...
    │   │   │   │   ├── rgb_front/
    │   │   │   │   │   ├── 0000.png
    │   │   │   │   │   └── ...
    │   │   │   │   ├── measurements/
    │   │   │   │   │   ├── 0000.json
    │   │   │   │   │   └── ...
    │   │   │   │   └── birdview/
    │   │   │   │       ├── 0000.png
    │   │   │   │       └── ...
    """
    
    def __init__(
        self,
        root_dir,
        towns=[5],
        weathers=[0],
        split='train',
        lidar_range=30,
        lidar_pixels_per_meter=7,
        image_size=256,
        return_raw_lidar=False,
        num_cameras=4
    ):
        """
        Args:
            root_dir: Root directory of V2Xverse dataset
            towns: List of town IDs to use
            weathers: List of weather IDs to use
            split: 'train', 'val', or 'test'
            lidar_range: Range for LiDAR BEV generation
            lidar_pixels_per_meter: Resolution for LiDAR BEV
            image_size: Size to resize all images to (square)
            return_raw_lidar: Whether to also return raw LiDAR points
            num_cameras: Number of camera views to load (1-4)
        """
        self.root_dir = Path(root_dir)
        self.towns = towns
        self.weathers = weathers
        self.split = split
        self.lidar_range = lidar_range
        self.lidar_pixels_per_meter = lidar_pixels_per_meter
        self.image_size = image_size
        self.return_raw_lidar = return_raw_lidar
        self.num_cameras = num_cameras
        
        # Image transforms 
        self.camera_image_transform = transforms.Compose([
            transforms.Resize(image_size),  
            transforms.CenterCrop((image_size, image_size)),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.bev_image_transform = transforms.Compose([
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.lidar_image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        # Build dataset index
        self.samples = self._build_index()
        
    def _build_index(self):
        """Build list of (route_path, frame_id) tuples."""
        samples = []
        
        for weather_id in self.weathers:
            weather_dir = self.root_dir / f"weather-{weather_id}" / "data"
            
            if not weather_dir.exists():
                print(f"Warning: Weather dir not found: {weather_dir}")
                continue
            
            # Find all routes
            for route_dir in sorted(weather_dir.glob("routes_*")):
                if not route_dir.is_dir():
                    continue
                
                # Extract town ID from route name (e.g., routes_town05_xxx -> 5)
                route_name = route_dir.name
                try:
                    town_id = int(route_name.split('town')[1][:2])
                except (IndexError, ValueError):
                    continue
                
                if town_id not in self.towns:
                    continue
                
                # Find ego vehicle directory
                ego_dir = route_dir / "ego_vehicle_0"
                if not ego_dir.exists():
                    continue
                
                # Find all frames
                lidar_dir = ego_dir / "lidar"
                if not lidar_dir.exists():
                    continue
                
                lidar_files = sorted(lidar_dir.glob("*.npy"))
                for lidar_file in lidar_files:
                    frame_id = int(lidar_file.stem)
                    samples.append((ego_dir, frame_id))
        
        print(f"Found {len(samples)} samples in {self.split} split")
        return samples
    
    def _load_lidar(self, ego_dir, frame_id):
        """Load LiDAR point cloud and generate BEV image."""
        lidar_path = ego_dir / "lidar" / f"{frame_id:04d}.npy"
        
        if not lidar_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
        
        # Load point cloud
        points = np.load(lidar_path)  # (N, 4): x, y, z, intensity
        
        # Generate BEV image from LiDAR
        bev_lidar = lidar_to_topview(
            points,
            range_=self.lidar_range,
            pixels_per_meter=self.lidar_pixels_per_meter
        )
        
        # Convert to tensor and resize
        bev_lidar_pil = Image.fromarray(bev_lidar)
        bev_lidar_tensor = self.lidar_image_transform(bev_lidar_pil)
        
        return bev_lidar_tensor, points if self.return_raw_lidar else None
    
    def _load_camera(self, ego_dir, cam_idx, frame_id):
        """Load camera images.
        cam_idx: 0=front, 1=right, 2=left, 3=rear"""
        
        if cam_idx == 0:
            camera_dir = ego_dir / f"rgb_front"
        elif cam_idx == 1:
            camera_dir = ego_dir / f"rgb_right"
        elif cam_idx == 2:
            camera_dir = ego_dir / f"rgb_left"
        elif cam_idx == 3:
            camera_dir = ego_dir / f"rgb_rear"

        camera_path = camera_dir / f"{frame_id:04d}.jpg"
        
        if not camera_path.exists():
            print(f"Warning: Camera image not found: {camera_path}")
            # Return black image if camera not found
            camera_image = Image.new('RGB', (self.image_size, self.image_size))
        else:
            camera_image = Image.open(camera_path).convert('RGB')
        
        camera_tensor = self.camera_image_transform(camera_image)
        
        return camera_tensor
    
    def _load_gt_bev(self, ego_dir, frame_id):
        """Load ground truth BEV image."""
        bev_path = ego_dir / "birdview" / f"{frame_id:04d}.jpg"
        
        if not bev_path.exists():
            print(f"Warning: GT BEV not found: {bev_path}")
            # Return black image if GT BEV not found
            gt_bev = Image.new('RGB', (self.image_size, self.image_size))
        else:
            gt_bev = Image.open(bev_path).convert('RGB')
        
        gt_bev_tensor = self.bev_image_transform(gt_bev)
        
        return gt_bev_tensor
    
    def _load_measurements(self, ego_dir, frame_id):
        """Load measurements (metadata) for the frame."""
        measurements_path = ego_dir / "measurements" / f"{frame_id:04d}.json"
        
        if measurements_path.exists():
            with open(measurements_path, 'r') as f:
                measurements = json.load(f)
        else:
            measurements = {}
        
        return measurements
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns dictionary with:
        - 'lidar_bev': LiDAR BEV image (3, H, W)
        - 'rgb_front': Camera image (3, H, W)
        - 'rgb_right': Camera image (3, H, W)
        - 'rgb_left': Camera image (3, H, W)
        - 'rgb_rear': Camera image (3, H, W)
        - 'gt_bev': Ground truth BEV image (3, H, W)
        - 'measurements': Metadata dict
        - 'frame_id': Frame ID
        - 'raw_lidar': Raw LiDAR points if return_raw_lidar=True
        """
        ego_dir, frame_id = self.samples[idx]
        
        # Load all data
        lidar_bev, raw_lidar = self._load_lidar(ego_dir, frame_id)
        rgb_front = self._load_camera(ego_dir, 0, frame_id)
        rgb_right = self._load_camera(ego_dir, 1, frame_id)
        rgb_left = self._load_camera(ego_dir, 2, frame_id)
        rgb_rear = self._load_camera(ego_dir, 3, frame_id)
        gt_bev = self._load_gt_bev(ego_dir, frame_id)
        measurements = self._load_measurements(ego_dir, frame_id)
        
        # Build output dictionary
        sample = {
            'lidar_bev': lidar_bev,          # (3, H, W)
            'rgb_front': rgb_front,          # (3, H, W)
            'rgb_right': rgb_right,          # (3, H, W)
            'rgb_left': rgb_left,            # (3, H, W)
            'rgb_rear': rgb_rear,            # (3, H, W)
            'gt_bev': gt_bev,                # (3, H, W)
            'measurements': measurements,     # dict
            'frame_id': frame_id,            # int
            'ego_dir': str(ego_dir),         # str
        }
        
        if self.return_raw_lidar and raw_lidar is not None:
            sample['raw_lidar'] = raw_lidar  # (N, 4)
        
        return sample


def create_bev_dataloader(
    root_dir,
    batch_size=8,
    num_workers=4,
    shuffle=True,
    towns=[5],
    weathers=[0],
    **dataset_kwargs
):
    """Convenience function to create a DataLoader."""
    dataset = BEVDataset(
        root_dir=root_dir,
        towns=towns,
        weathers=weathers,
        **dataset_kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    dataset = BEVDataset(
        root_dir="external_paths/data_root",
        towns=[5],
        weathers=[0],
        image_size=256,
        return_raw_lidar=False,
        num_cameras=4
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load a sample
    sample = dataset[100]
    
    print("\nSample contents:")
    print(f"  lidar_bev shape: {sample['lidar_bev'].shape}")
    print(f"  rgb_front shape: {sample['rgb_front'].shape}")
    print(f"  rgb_right shape: {sample['rgb_right'].shape}")
    print(f"  rgb_left shape: {sample['rgb_left'].shape}")
    print(f"  rgb_rear shape: {sample['rgb_rear'].shape}")
    print(f"  gt_bev shape: {sample['gt_bev'].shape}")
    print(f"  frame_id: {sample['frame_id']}")
    print(f"  measurements keys: {list(sample['measurements'].keys())}")
    
    # Visualize
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # LiDAR BEV
    axes[0].imshow(sample['lidar_bev'].permute(1, 2, 0))
    axes[0].set_title("LiDAR BEV")
    axes[0].axis('off')
    
    # First camera
    axes[1].imshow(sample['rgb_front'].permute(1, 2, 0))
    axes[1].set_title("RGB Front")
    axes[1].axis('off')
    
    # GT BEV
    axes[2].imshow(sample['gt_bev'].permute(1, 2, 0))
    axes[2].set_title("GT BEV")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("bev_generator/bev_dataset_sample.png", dpi=150)
    print("\nSaved visualization to bev_dataset_sample.png")
