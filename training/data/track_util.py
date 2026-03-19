"""
Track utilities for epipolar geometry validation.

This module provides functions for validating track quality using epipolar geometry
constraints, which helps ensure that point correspondences across frames are geometrically
consistent.
"""

import numpy as np
import torch
from typing import Tuple


def track_epipolar_check(tracks: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, 
                        use_essential_mat: bool = False) -> torch.Tensor:
    """
    Validate track quality using epipolar geometry constraints.
    
    Computes Sampson epipolar distance for tracks across frames to validate
    geometric consistency. Tracks that violate epipolar constraints are
    likely to be incorrect correspondences.
    
    Args:
        tracks: Tensor of shape (B, T, 2) containing track coordinates (x, y)
        extrinsics: Tensor of shape (B, 3, 4) containing camera extrinsic matrices
        intrinsics: Tensor of shape (B, 3, 3) containing camera intrinsic matrices
        use_essential_mat: If True, use essential matrix (normalized coordinates),
                          if False, use fundamental matrix (pixel coordinates)
    
    Returns:
        torch.Tensor: Sampson distances of shape (B-1, T) for track validation
    """
    from kornia.geometry.epipolar import sampson_epipolar_distance

    B, T, _ = tracks.shape
    
    # Compute essential matrices between first frame and all other frames
    essential_mats = get_essential_matrix(
        extrinsics[0:1].expand(B-1, -1, -1), 
        extrinsics[1:]
    )

    if use_essential_mat:
        # Use essential matrix with normalized coordinates
        tracks_normalized = cam_from_img(tracks, intrinsics)
        sampson_distances = sampson_epipolar_distance(
            tracks_normalized[0:1].expand(B-1, -1, -1), 
            tracks_normalized[1:], 
            essential_mats
        )
    else:
        # Use fundamental matrix with pixel coordinates
        K1 = intrinsics[0:1].expand(B-1, -1, -1)
        K2 = intrinsics[1:].expand(B-1, -1, -1)
        fundamental_mats = K2.inverse().permute(0, 2, 1).matmul(essential_mats).matmul(K1.inverse())
        sampson_distances = sampson_epipolar_distance(
            tracks[0:1].expand(B-1, -1, -1), 
            tracks[1:], 
            fundamental_mats
        )

    return sampson_distances


def get_essential_matrix(extrinsic1: torch.Tensor, extrinsic2: torch.Tensor) -> torch.Tensor:
    """
    Compute essential matrix between two camera poses.
    
    The essential matrix encodes the epipolar geometry between two calibrated cameras.
    It relates corresponding points in normalized image coordinates.
    
    Args:
        extrinsic1: Tensor of shape (B, 3, 4) - first camera extrinsic matrix
        extrinsic2: Tensor of shape (B, 3, 4) - second camera extrinsic matrix
    
    Returns:
        torch.Tensor: Essential matrices of shape (B, 3, 3)
    """
    R1 = extrinsic1[:, :3, :3]
    t1 = extrinsic1[:, :3, 3]
    R2 = extrinsic2[:, :3, :3]
    t2 = extrinsic2[:, :3, 3]
    
    # Relative rotation and translation
    R12 = R2.matmul(R1.permute(0, 2, 1))
    t12 = t2 - R12.matmul(t1[..., None])[..., 0]
    
    # Essential matrix: E = [t]_× R
    E_R = R12
    E_t = -E_R.permute(0, 2, 1).matmul(t12[..., None])[..., 0]
    E = E_R.matmul(hat(E_t))
    
    return E


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the skew-symmetric matrix (hat operator) for cross product.
    
    For a 3D vector v = [x, y, z], returns the skew-symmetric matrix:
    [[ 0, -z,  y],
     [ z,  0, -x],
     [-y,  x,  0]]
    
    This matrix satisfies: hat(v) @ u = v × u (cross product)
    
    Args:
        v: Tensor of shape (N, 3) containing 3D vectors
    
    Returns:
        torch.Tensor: Skew-symmetric matrices of shape (N, 3, 3)
    """
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    x, y, z = v.unbind(1)

    h_01 = -z.view(N, 1, 1)
    h_02 = y.view(N, 1, 1)
    h_10 = z.view(N, 1, 1)
    h_12 = -x.view(N, 1, 1)
    h_20 = -y.view(N, 1, 1)
    h_21 = x.view(N, 1, 1)

    zeros = torch.zeros((N, 1, 1), dtype=v.dtype, device=v.device)

    row1 = torch.cat((zeros, h_01, h_02), dim=2)
    row2 = torch.cat((h_10, zeros, h_12), dim=2)
    row3 = torch.cat((h_20, h_21, zeros), dim=2)

    h = torch.cat((row1, row2, row3), dim=1)

    return h


def cam_from_img(tracks: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Convert image coordinates to normalized camera coordinates.
    
    Transforms pixel coordinates to normalized camera coordinates using
    the camera intrinsic matrix. This is useful for epipolar geometry
    computations with essential matrices.
    
    Args:
        tracks: Tensor of shape (..., 2) containing pixel coordinates (x, y)
        intrinsics: Tensor of shape (..., 3, 3) containing camera intrinsic matrices
    
    Returns:
        torch.Tensor: Normalized camera coordinates of the same shape as tracks
    """
    # Get intrinsic parameters
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1] 
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    
    # Convert to normalized coordinates
    x_norm = (tracks[..., 0] - cx) / fx
    y_norm = (tracks[..., 1] - cy) / fy
    
    return torch.stack([x_norm, y_norm], dim=-1)


def validate_tracks_epipolar(tracks: torch.Tensor, extrinsics: torch.Tensor, 
                           intrinsics: torch.Tensor, threshold: float = 5.0) -> torch.Tensor:
    """
    Validate tracks using epipolar geometry and return a quality mask.
    
    Args:
        tracks: Tensor of shape (B, T, 2) containing track coordinates
        extrinsics: Tensor of shape (B, 3, 4) containing camera extrinsic matrices
        intrinsics: Tensor of shape (B, 3, 3) containing camera intrinsic matrices
        threshold: Maximum allowed Sampson distance in pixels
    
    Returns:
        torch.Tensor: Boolean mask of shape (T,) indicating valid tracks
    """
    # Compute Sampson distances
    sampson_distances = track_epipolar_check(tracks, extrinsics, intrinsics, use_essential_mat=False)
    
    # Check if all frame pairs satisfy the epipolar constraint
    # Shape: (B-1, T) -> (T,) by checking if all frame pairs are below threshold
    epipolar_valid = (sampson_distances < threshold).all(dim=0)
    
    return epipolar_valid


def filter_tracks_by_epipolar(tracks: torch.Tensor, track_vis_mask: torch.Tensor,
                             extrinsics: torch.Tensor, intrinsics: torch.Tensor,
                             threshold: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter tracks based on epipolar geometry validation.
    
    Args:
        tracks: Tensor of shape (B, T, 2) containing track coordinates
        track_vis_mask: Tensor of shape (B, T) indicating track visibility
        extrinsics: Tensor of shape (B, 3, 4) containing camera extrinsic matrices  
        intrinsics: Tensor of shape (B, 3, 3) containing camera intrinsic matrices
        threshold: Maximum allowed Sampson distance in pixels
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Filtered tracks and visibility masks
    """
    # Get epipolar validation mask
    epipolar_valid = validate_tracks_epipolar(tracks, extrinsics, intrinsics, threshold)
    
    # Filter tracks and visibility masks
    valid_tracks = tracks[:, epipolar_valid]
    valid_vis_mask = track_vis_mask[:, epipolar_valid]
    
    return valid_tracks, valid_vis_mask 