#!/usr/bin/env python3
"""
Simplified BEVFormer: LiDAR BEV query + Camera features with cross-attention
Output: Pixel-wise BEV segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CameraBackbone(nn.Module):
    """ResNet backbone for extracting camera features."""
    
    def __init__(self, pretrained=True, out_channels=256):
        super().__init__()
        
        # Load ResNet backbone
        backbone = models.resnet50(pretrained=pretrained)
        in_channels = 2048
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Projection layer to match output channels
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - camera image
        Returns:
            features: (B, out_channels, H/32, W/32) - extracted features
        """
        features = self.backbone(x)  # (B, in_channels, H/32, W/32)
        features = self.projection(features)  # (B, out_channels, H/32, W/32)
        return features


class LiDAREncoder(nn.Module):
    """Encoder for LiDAR BEV to extract query features."""
    
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - LiDAR BEV image
        Returns:
            query: (B, out_channels, H/4, W/4) - query features
        """
        x = self.conv1(x)  # (B, 64, H/2, W/2)
        x = self.conv2(x)  # (B, 128, H/4, W/4)
        x = self.conv3(x)  # (B, out_channels, H/4, W/4)
        return x


class CrossAttention(nn.Module):
    """Cross-attention layer for fusing LiDAR query with camera features."""
    
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Query from LiDAR BEV
        self.query_proj = nn.Linear(dim, dim)
        # Key and Value from camera features
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, query, key_value):
        """
        Args:
            query: (B, C, H_q, W_q) - from LiDAR BEV encoder
            key_value: (B, C, H_kv, W_kv) - from camera backbones
        Returns:
            output: (B, C, H_q, W_q) - cross-attended features
        """
        B, C, H_q, W_q = query.shape
        _, _, H_kv, W_kv = key_value.shape
        
        # Flatten spatial dimensions
        query_flat = query.flatten(2).permute(0, 2, 1)  # (B, H_q*W_q, C)
        key_flat = key_value.flatten(2).permute(0, 2, 1)  # (B, H_kv*W_kv, C)
        value_flat = key_value.flatten(2).permute(0, 2, 1)  # (B, H_kv*W_kv, C)
        
        # Project to multi-head format
        Q = self.query_proj(query_flat)  # (B, H_q*W_q, C)
        K = self.key_proj(key_flat)  # (B, H_kv*W_kv, C)
        V = self.value_proj(value_flat)  # (B, H_kv*W_kv, C)
        
        # Reshape for multi-head attention
        Q = Q.reshape(B, H_q * W_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (B, num_heads, H_q*W_q, head_dim)
        K = K.reshape(B, H_kv * W_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (B, num_heads, H_kv*W_kv, head_dim)
        V = V.reshape(B, H_kv * W_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # (B, num_heads, H_kv*W_kv, head_dim)
        
        # Compute attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, num_heads, H_q*W_q, H_kv*W_kv)
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = attn @ V  # (B, num_heads, H_q*W_q, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, H_q * W_q, C)  # (B, H_q*W_q, C)
        out = self.out_proj(out)  # (B, H_q*W_q, C)
        
        # Reshape back to spatial
        out = out.permute(0, 2, 1).reshape(B, C, H_q, W_q)  # (B, C, H_q, W_q)
        
        return out


class BEVDecoder(nn.Module):
    """BEV decoder to generate pixel-wise segmentation."""
    
    def __init__(self, in_channels=256, num_classes=1):
        super().__init__()
        
        # Upsample to original resolution
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H/4, W/4) - cross-attended features
        Returns:
            bev_pred: (B, num_classes, H, W) - pixel-wise BEV prediction
        """
        x = self.up1(x)  # (B, 128, H/2, W/2)
        x = self.up2(x)  # (B, 64, H, W)
        x = self.seg_head(x)  # (B, num_classes, H, W)
        return x


class SimpleBEVFormer(nn.Module):
    """
    Simplified BEVFormer for pixel-wise BEV segmentation.
    
    Architecture:
    1. LiDAR BEV -> Encoder -> Query features
    2. Camera images -> Backbones -> Key/Value features
    3. Cross-Attention: Query attends to camera features
    4. BEV Decoder -> Pixel-wise prediction
    """
    
    def __init__(
        self,
        num_cameras=2,
        in_channels=256,
        num_classes=3,
        pretrained=True
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        
        # LiDAR encoder: LiDAR BEV (3 channels) -> query features
        self.lidar_encoder = LiDAREncoder(in_channels=3, out_channels=in_channels)
        
        # Camera backbones: Extract features from each camera
        self.camera_backbones = nn.ModuleList([
            CameraBackbone( pretrained=pretrained, 
                          out_channels=in_channels)
            for _ in range(num_cameras)
        ])
        
        # Cross-attention: Fuse LiDAR query with camera features
        self.cross_attention = CrossAttention(dim=in_channels, num_heads=8)
        
        # Feature fusion: Combine attended features with original query
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # BEV decoder: Generate pixel-wise segmentation
        self.bev_decoder = BEVDecoder(in_channels=in_channels, num_classes=num_classes)
    
    def forward(self, lidar_bev, camera_images):
        """
        Args:
            lidar_bev: (B, 3, H, W) - LiDAR top-view image
            camera_images: (B, num_cameras, 3, H, W) - Multi-view camera images
        
        Returns:
            bev_pred: (B, num_classes, H, W) - Pixel-wise BEV segmentation
        """
        B, C, H, W = lidar_bev.shape
        
        # Extract query features from LiDAR BEV
        query = self.lidar_encoder(lidar_bev)  # (B, in_channels, H/4, W/4)
        
        # Extract features from each camera and fuse them
        camera_features_list = []
        for cam_idx in range(self.num_cameras):
            cam_img = camera_images[:, cam_idx]  # (B, 3, H, W)
            cam_feat = self.camera_backbones[cam_idx](cam_img)  # (B, in_channels, H/32, W/32)
            # Upsample camera features to match query resolution
            cam_feat = F.interpolate(cam_feat, size=query.shape[-2:], mode='bilinear', align_corners=False)
            camera_features_list.append(cam_feat)
        
        # Fuse all camera features by averaging
        camera_features_fused = torch.stack(camera_features_list, dim=0).mean(dim=0)  # (B, in_channels, H/4, W/4)
        
        # Cross-attention: LiDAR query attends to camera features
        attended_features = self.cross_attention(query, camera_features_fused)  # (B, in_channels, H/4, W/4)
        
        # Fusion: Combine attended features with original query
        fused = torch.cat([query, attended_features], dim=1)  # (B, 2*in_channels, H/4, W/4)
        fused = self.fusion(fused)  # (B, in_channels, H/4, W/4)
        
        # BEV decoder: Generate pixel-wise prediction
        bev_pred = self.bev_decoder(fused)  # (B, num_classes, H, W)
        
        return bev_pred


if __name__ == "__main__":
    import torch
    
    # Test the model
    batch_size = 2
    num_cameras = 2
    height, width = 256, 256
    
    # Create model
    model = SimpleBEVFormer(num_cameras=num_cameras, num_classes=3)
    
    # Create dummy inputs
    lidar_bev = torch.randn(batch_size, 3, height, width)
    camera_images = torch.randn(batch_size, num_cameras, 3, height, width)
    
    # Forward pass
    output = model(lidar_bev, camera_images)
    
    print(f"Input LiDAR BEV shape: {lidar_bev.shape}")
    print(f"Input camera images shape: {camera_images.shape}")
    print(f"Output BEV segmentation shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
