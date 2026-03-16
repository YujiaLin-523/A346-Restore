import os
import yaml
import numpy as np
import cv2
import re

def load_config(config_path="config/config.yaml"):
    """Load configuration from yaml file."""
    # Resolve absolute path relative to the workspace root if needed, 
    # but here we assume running from root or config_path is relative to cwd.
    if not os.path.exists(config_path):
        # Try finding it relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_matrix(path):
    """Load 4x4 matrix from a text file with space-separated values."""
    if not os.path.exists(path):
        # Try finding it relative to workspace root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Matrix file not found: {path}")

    with open(path, 'r') as f:
        txt = f.read()
    
    # Extract all numbers using regex
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", txt)]
    if len(nums) < 16:
        raise ValueError(f"File {path} does not contain enough numbers for 4x4 matrix")
    
    return np.array(nums[:16], dtype=np.float64).reshape(4, 4)

def save_matrix(matrix, path):
    """Save 4x4 matrix to text file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, path) if not os.path.isabs(path) else path
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, "w") as f:
        for r in range(4):
            f.write(" ".join([f"{matrix[r,c]:.8f}" for c in range(4)]) + "\n")
    print(f"[INFO] Saved matrix to {full_path}")

def make_T(R, t):
    """Create 4x4 homogeneous transform from 3x3 R and 3x1 t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def get_robust_depth(depth_img, u, v, roi_size=5, depth_scale=0.001):
    """
    Get robust depth at (u,v) by taking median of neighborhood.
    Handles 0s and NaNs.
    """
    if depth_img is None:
        return None
        
    h, w = depth_img.shape[:2]
    win = roi_size // 2
    u0, u1 = max(0, u-win), min(w-1, u+win)
    v0, v1 = max(0, v-win), min(h-1, v+win)
    
    patch = depth_img[v0:v1+1, u0:u1+1].copy()
    
    # Mask invalid values (0 or inf/nan)
    valid_mask = (patch > 0) & np.isfinite(patch)
    vals = patch[valid_mask].astype(np.float64)
    
    if vals.size == 0:
        return None
        
    # Median is robust to outliers
    z = np.median(vals) * depth_scale
    return float(z)

def pixel_to_3d(u, v, z, K, D=None):
    """
    Back-project pixel (u,v) with depth z to 3D point in camera frame.
    """
    if z is None or z <= 0:
        return None

    if D is None or np.allclose(D, 0):
        # Pinhole model
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z])
    else:
        # Undistort point first
        # cv2.undistortPoints returns normalized coordinates
        pts = np.array([[[u, v]]], dtype=np.float64)
        norm_pts = cv2.undistortPoints(pts, K, D)
        x_norm, y_norm = norm_pts[0,0]
        x = x_norm * z
        y = y_norm * z
        return np.array([x, y, z])

def transform_point(T, point):
    """Apply 4x4 transform to 3D point."""
    p4 = np.append(point, 1.0)
    res = T @ p4
    return res[:3]
