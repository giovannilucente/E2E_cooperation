#!/usr/bin/env python3
"""
Convert LiDAR point cloud to top-view (BEV) image.

Usage:
    python lidar_topview.py <lidar_file> [output_image] [range] [resolution]

Example:
    python lidar_topview.py data/lidar.npy output.png 50 5
    python lidar_topview.py data/lidar.npy  # Uses defaults
"""

import numpy as np
import cv2
import sys
from pathlib import Path


def load_lidar_file(lidar_path):
    """Load lidar point cloud from .npy or .bin file."""
    lidar_path = Path(lidar_path)
    
    if lidar_path.suffix == '.npy':
        points = np.load(lidar_path)
    elif lidar_path.suffix == '.bin':
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    else:
        raise ValueError(f"Unsupported file format: {lidar_path.suffix}")
    
    return points


def lidar_to_topview(
    lidar_points,
    range_=50,
    pixels_per_meter=5,
    intensity_channel=3,
    height_min=-5,
    height_max=5
):
    """
    Project 3D lidar point cloud to 2D top-view BEV image.
    
    Args:
        lidar_points: (N, 4) array with [x, y, z, intensity]
        range_: Detection range in meters (e.g., 50m)
        pixels_per_meter: Resolution in pixels per meter
        intensity_channel: Which channel to use for intensity (3 for typical lidar)
        height_min: Filter points below this height
        height_max: Filter points above this height
    
    Returns:
        image: (H, W, 3) uint8 BEV image
    """
    
    # Image dimensions
    img_size = int(range_ * 2 * pixels_per_meter)
    image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Extract coordinates
    x = lidar_points[:, 0]
    y = lidar_points[:, 1]
    z = lidar_points[:, 2]
    
    # Get intensity (4th channel), clamp between 0-1
    if lidar_points.shape[1] >= 4:
        intensity = lidar_points[:, intensity_channel]
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    else:
        intensity = np.ones_like(x)
    
    # Filter by height
    height_mask = (z >= height_min) & (z <= height_max)
    x = x[height_mask]
    y = y[height_mask]
    intensity = intensity[height_mask]
    
    # Filter by range
    range_mask = (x**2 + y**2) <= (range_**2)
    x = x[range_mask]
    y = y[range_mask]
    intensity = intensity[range_mask]
    
    # Convert to image coordinates
    # x-axis goes right, y-axis goes back (away from ego)
    # Origin is at image center
    img_x = (- x + range_) / (range_ * 2) * img_size
    img_y = (- y + range_) / (range_ * 2) * img_size
    
    # Draw points on image
    for i in range(len(x)):
        px = int(np.clip(img_x[i], 0, img_size - 1))
        py = int(np.clip(img_y[i], 0, img_size - 1))
        
        # Color: intensity in grayscale
        color_value = int(intensity[i] * 255)
        image[py, px] = [color_value, color_value, color_value]
    
    return image


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    lidar_file = sys.argv[1]
    lidar_file = "external_paths/data_root/weather-0/data/routes_town01_long_w0_04_29_17_35_41/ego_vehicle_0/lidar/0000.npy"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "bev_generator/lidar_topview.png"
    range_ = float(sys.argv[3]) if len(sys.argv) > 3 else 30
    pixels_per_meter = float(sys.argv[4]) if len(sys.argv) > 4 else 7
    
    print(f"Loading LiDAR file: {lidar_file}")
    points = load_lidar_file(lidar_file)
    print(f"Loaded {len(points)} points")
    print(f"Point cloud shape: {points.shape}")
    print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    print(f"\nGenerating top-view image...")
    print(f"  Range: {range_}m")
    print(f"  Resolution: {pixels_per_meter} pixels/meter")
    
    image = lidar_to_topview(
        points,
        range_=range_,
        pixels_per_meter=pixels_per_meter
    )
    
    cv2.imwrite(output_file, image)
    print(f"✓ Saved to: {output_file}")
    print(f"  Image size: {image.shape[0]}x{image.shape[1]} pixels")


if __name__ == "__main__":
    main()
