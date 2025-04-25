import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def calculate_hd95(pred, target, voxel_spacing=1):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    pred_coords = np.transpose(np.nonzero(pred_np))
    target_coords = np.transpose(np.nonzero(target_np))

    pred_coords_scaled = pred_coords * voxel_spacing
    target_coords_scaled = target_coords * voxel_spacing

    hd_distance, _, _ = directed_hausdorff(pred_coords_scaled, target_coords_scaled)

    hd95 = np.percentile(hd_distance, 95)

    return hd95

def iou_score(pred, target):
    smooth = 1e-5

    output = pred.numpy()
    target = target.numpy()

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def dice_score(pred, targs):
    pred = (pred > 0).float()
    return (2. * (pred*targs).sum() / (pred+targs).sum()).item()

def dice_coef2(pred, target):
    "This metric is for validation purpose"
    smooth = 1e-5

    pred = (pred>0.5).numpy()
    target = target.numpy()
    intersection = (pred * target).sum()

    return (2. * intersection + smooth) / \
        (pred.sum() + target.sum() + smooth)