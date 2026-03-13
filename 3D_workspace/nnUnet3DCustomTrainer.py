import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
from datetime import datetime
import kornia
import random

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from .custom_loss import DC_and_CE_with_Edge_Loss, DeepSupervisionWrapper


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class GPU3DSpatialAugmentation(nn.Module):
    def __init__(self, 
                 patch_size, 
                 # Rotation: (Pitch, Yaw, Roll)
                 rot_range_deg=(0, 90, 0), 
                 
                 scale_range=(0.7, 1.7),
                 
                 # Translation: Fraction of dimension (0.0 - 1.0)
                 trans_range=(0.45, 0.45, 0.45),
                 
                 # Shear: (xy, xz, yx, yz, zx, zy)
                 shear_range_deg=(0, 15, 0, 0, 15, 0),
                 
                 persp_factor=0.35, 
                 
                 # This prevents the "fan" effect and keeps slices parallel.
                 keep_y_parallel=True,
                 
                 p_affine=0.95, 
                 p_flip=0.5):
        super().__init__()
        
        self.patch_size = patch_size
        self.rot_range_rad = [np.deg2rad(r) for r in rot_range_deg]
        self.scale_range = scale_range
        self.trans_range = trans_range
        self.shear_range_rad = [np.deg2rad(s) for s in shear_range_deg]
        
        # Scaling perspective factor down for stability
        self.persp_factor = persp_factor * 0.1 
        self.keep_y_parallel = keep_y_parallel
        
        self.p_affine = p_affine
        self.p_flip = p_flip

    def _get_transform_matrix(self, B, device):
        # 1. Scaling
        s_factors = torch.rand(B, 3, device=device) * \
                    (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        scale_mat = torch.diag_embed(torch.cat([s_factors, torch.ones(B, 1, device=device)], dim=1))

        # 2. Rotation
        rand_rad = lambda r: (torch.rand(B, device=device) * 2 - 1) * r if r > 0 else torch.zeros(B, device=device)
        rx, ry, rz = rand_rad(self.rot_range_rad[0]), rand_rad(self.rot_range_rad[1]), rand_rad(self.rot_range_rad[2])
        
        zeros = torch.zeros_like(rx)
        ones = torch.ones_like(rx)

        # Rot X
        rot_x = torch.stack([ones, zeros, zeros, zeros, zeros, rx.cos(), -rx.sin(), zeros, zeros, rx.sin(), rx.cos(), zeros, zeros, zeros, zeros, ones], dim=1).view(B, 4, 4)
        # Rot Y (Your Slice Rotation)
        rot_y = torch.stack([ry.cos(), zeros, ry.sin(), zeros, zeros, ones, zeros, zeros, -ry.sin(), zeros, ry.cos(), zeros, zeros, zeros, zeros, ones], dim=1).view(B, 4, 4)
        # Rot Z
        rot_z = torch.stack([rz.cos(), -rz.sin(), zeros, zeros, rz.sin(), rz.cos(), zeros, zeros, zeros, zeros, ones, zeros, zeros, zeros, zeros, ones], dim=1).view(B, 4, 4)

        rot_mat = rot_z @ rot_y @ rot_x

        # 3. Shear
        def get_s(idx):
            r = self.shear_range_rad[idx]
            if r <= 1e-6: return torch.zeros(B, device=device)
            return (torch.rand(B, device=device) * 2 - 1) * np.tan(r)

        S_xy, S_xz = get_s(0), get_s(1)
        S_yx, S_yz = get_s(2), get_s(3) # Should be 0 to protect Y
        S_zx, S_zy = get_s(4), get_s(5)

        shear_mat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        shear_mat[:, 0, 1] = S_xy
        shear_mat[:, 0, 2] = S_xz
        shear_mat[:, 1, 0] = S_yx
        shear_mat[:, 1, 2] = S_yz
        shear_mat[:, 2, 0] = S_zx
        shear_mat[:, 2, 1] = S_zy

        # 4. Translation
        # Range is relative to [-1, 1], so we multiply by 2.0 to span full grid
        tx = ((torch.rand(B, device=device) * 2 - 1) * self.trans_range[0]) * 2.0
        ty = ((torch.rand(B, device=device) * 2 - 1) * self.trans_range[1]) * 2.0
        tz = ((torch.rand(B, device=device) * 2 - 1) * self.trans_range[2]) * 2.0

        trans_mat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        trans_mat[:, 0, 3] = tx
        trans_mat[:, 1, 3] = ty
        trans_mat[:, 2, 3] = tz

        # 5. Perspective
        persp_mat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
        if self.persp_factor > 0:
            p_coeffs = (torch.rand(B, 3, device=device) * 2 - 1) * self.persp_factor
            persp_mat[:, 3, 0] = p_coeffs[:, 0]
            # Even if we "parallelize" Y later, keeping this 0 helps stability
            persp_mat[:, 3, 1] = 0.0 
            persp_mat[:, 3, 2] = p_coeffs[:, 2]

        return persp_mat @ trans_mat @ shear_mat @ rot_mat @ scale_mat

    def _create_perfect_grid(self, shape, device):
        B, C, D, H, W = shape
        def make_coord(size):
            return (torch.arange(size, device=device, dtype=torch.float32) + 0.5) / size * 2 - 1
        
        # ij indexing -> (D, H, W) order -> (z, y, x)
        zs, ys, xs = make_coord(D), make_coord(H), make_coord(W)
        mesh_z, mesh_y, mesh_x = torch.meshgrid(zs, ys, xs, indexing='ij')
        
        # Grid: (x, y, z, 1)
        grid = torch.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten(), torch.ones_like(mesh_x.flatten())], dim=0)
        return grid.unsqueeze(0).expand(B, -1, -1)

    def forward(self, data, target):
        B, C, D, H, W = data.shape
        device = data.device

        # 1. Flip
        if torch.rand(1) < self.p_flip:
            dims_to_try = [2, 4] # Flip Depth or Width, preserve Y
            for d in dims_to_try:
                if random.random() < 0.5:
                    data = torch.flip(data, [d])
                    target = torch.flip(target, [d])

        # 2. Affine/Persp
        if random.random() < self.p_affine:
            fwd_mat = self._get_transform_matrix(B, device)
            
            try:
                inv_mat = torch.linalg.inv(fwd_mat)
            except RuntimeError:
                inv_mat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)

            grid_flat = self._create_perfect_grid(data.shape, device)
            grid_trans = torch.bmm(inv_mat, grid_flat) # (B, 4, N)

            # Perspective Division with Y-Protection
            w_t = grid_trans[:, 3, :]
            w_t = torch.clamp(w_t, min=1e-4) # Avoid zero div
            
            x_t = grid_trans[:, 0, :] / w_t
            z_t = grid_trans[:, 2, :] / w_t
            
            if self.keep_y_parallel:
                # Cylindrical Perspective: Y is NOT scaled by depth (w)
                # It effectively ignores the perspective shrinking, keeping slices parallel.
                # Note: We still use the transformed Y coordinate (so rotation/shear applies), 
                # we just don't compress it based on Z-depth.
                y_t = grid_trans[:, 1, :] 
            else:
                # Standard Homography (Y shrinks at depth)
                y_t = grid_trans[:, 1, :] / w_t
            
            grid = torch.stack([x_t, y_t, z_t], dim=2).reshape(B, D, H, W, 3)

            data = F.grid_sample(data, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            target = F.grid_sample(target.float(), grid, mode='nearest', padding_mode='zeros', align_corners=False).long()

        return data, target
    

# --- 3D GPU PREPROCESSING FUNCTION ---

def _preprocess_data_gpu_3d(data_batch: torch.Tensor, mag_prepro: bool = False, phase_prepro: bool = False) -> torch.Tensor:
    was_5dim = False
    if len(data_batch.size()) == 4: 
        was_5dim = True
        data_batch = data_batch.unsqueeze(0)
    
    b, c, d, h, w = data_batch.shape
    mag_batch = data_batch[:, 0:1, :, :, :] 
    
    # 1. 3D Otsu
    mag_flattened = mag_batch.view(b, 1, d * h, w)
    thresholds_tensor = kornia.filters.otsu_threshold(mag_flattened)[1]
    reshaped_thresholds = thresholds_tensor.view(b, 1, 1, 1, 1)
    mask = (mag_batch >= reshaped_thresholds).float()

    # 2. 3D Morphology
    kernel_size = 3
    padding = 1
    eroded_mask = -F.max_pool3d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    opened_mask = F.max_pool3d(eroded_mask, kernel_size=kernel_size, stride=1, padding=padding)
    mask = opened_mask

    processed_mag = mag_batch * mask
    processed_phase = None
    if c > 1:
        phase_batch = data_batch[:, 1:2, :, :, :]
        processed_phase = phase_batch * mask

    # 3. Magnitude Preprocessing
    if mag_prepro:
        mag_min = processed_mag.amin(dim=(-3, -2, -1), keepdim=True)
        mag_max = processed_mag.amax(dim=(-3, -2, -1), keepdim=True)
        mag_normalized = (processed_mag - mag_min) / (mag_max - mag_min + 1e-6)
        
        mag_slices = mag_normalized.view(b * d, 1, h, w)
        mag_clahe_slices = kornia.enhance.equalize_clahe(mag_slices)
        mag_clahe = mag_clahe_slices.view(b, 1, d, h, w)
        processed_mag = mag_clahe * mask

    # 4. Phase Preprocessing
    if phase_prepro and c > 1:
        safe_mask = mask.bool()
        if safe_mask.any():
            masked_phase_values = torch.masked_select(phase_batch, safe_mask)
            p30 = torch.quantile(masked_phase_values, 0.30)
            p85 = torch.quantile(masked_phase_values, 0.85)
        else:
            p30, p85 = 0.0, 1.0
        phase_rescaled = (phase_batch - p30) / (p85 - p30 + 1e-6)
        phase_rescaled = torch.clamp(phase_rescaled, 0, 1)
        processed_phase = phase_rescaled * mask
    
    if c > 1:
        processed_batch = torch.cat([processed_mag, processed_phase], dim=1)
    else:
        processed_batch = processed_mag
        
    if was_5dim: processed_batch = processed_batch.squeeze(0)
    return processed_batch

# --- VISUALIZATION HELPERS ---

def decode_bitmask_to_multichannel(bitmask: np.ndarray, num_classes: int) -> list:
    """
    Decodes a label map into a list of binary masks per class.
    """
    masks = []
    # Assuming 0 is background, start from 1
    for i in range(1, num_classes + 1):
        masks.append(bitmask == i)
    return masks

import matplotlib.pyplot as plt
import numpy as np

def plot_3d_snapshot(mag_vol, phase_vol, gt_vol, pred_vol, num_classes, epoch, 
                     title_prefix="Validation", edge_vol=None):
    """
    Plots Axial, Coronal, and Sagittal views.
    Includes an optional 5th column for Edge/Weight Map if 'edge_vol' is provided.
    """
    shp = mag_vol.shape
    mid_z, mid_y, mid_x = shp[0] // 2, shp[1] // 2, shp[2] // 2
    
    views = [
        ('Coronal', lambda x: x[mid_z, :, :]), 
        ('Axial', lambda x: x[:, mid_y, :]), 
        ('Sagittal', lambda x: x[:, :, mid_x])
    ]
    
    n_cols = 5 if edge_vol is not None else 4
    
    fig, axs = plt.subplots(3, n_cols, figsize=(5 * n_cols, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.suptitle(f'{title_prefix} Snapshot - Epoch {epoch}', fontsize=16)
    
    colors = ['cyan', 'lime', 'red', 'yellow', 'magenta', 'orange', 'purple']

    def decode_bitmask_to_multichannel(bitmask, num_classes):
        masks = []
        for i in range(1, num_classes + 1):
            masks.append(bitmask == i)
        return masks

    for row_idx, (view_name, slicer) in enumerate(views):
        # 1. Magnitude
        axs[row_idx, 0].imshow(slicer(mag_vol).T, cmap='gray')
        axs[row_idx, 0].axis('off')
        axs[row_idx, 0].set_title(f"{view_name} - Mag")

        # 2. Phase
        if np.count_nonzero(phase_vol) == 0: 
            axs[row_idx, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axs[row_idx, 1].axis('off')
        else: 
            axs[row_idx, 1].imshow(slicer(phase_vol).T, cmap='gray')
            axs[row_idx, 1].axis('off')
        axs[row_idx, 1].set_title(f"{view_name} - Phase")

        # 3. Ground Truth
        axs[row_idx, 2].imshow(slicer(mag_vol).T, cmap='gray') 
        gt_slice = slicer(gt_vol)
        gt_channels = decode_bitmask_to_multichannel(gt_slice, num_classes)
        for i, channel in enumerate(gt_channels):
            if np.any(channel):
                color = colors[i % len(colors)]
                axs[row_idx, 2].contourf(channel.T, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                axs[row_idx, 2].contour(channel.T, levels=[0.5], colors=[color], linewidths=1)
        axs[row_idx, 2].axis('off')
        axs[row_idx, 2].set_title(f"{view_name} - GT")

        # 4. Prediction
        axs[row_idx, 3].imshow(slicer(mag_vol).T, cmap='gray')
        pred_slice = slicer(pred_vol)
        pred_channels = decode_bitmask_to_multichannel(pred_slice, num_classes)
        for i, channel in enumerate(pred_channels):
            if np.any(channel):
                color = colors[i % len(colors)]
                axs[row_idx, 3].contourf(channel.T, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                axs[row_idx, 3].contour(channel.T, levels=[0.5], colors=[color], linewidths=1)
        axs[row_idx, 3].axis('off')
        axs[row_idx, 3].set_title(f"{view_name} - Pred")
        
        # 5. Edge / Weight Map (FIXED LOGIC)
        if edge_vol is not None and n_cols > 4:
            # A. Plot Background
            axs[row_idx, 4].imshow(slicer(mag_vol).T, cmap='gray')
            
            # B. Get Slice
            edge_slice = slicer(edge_vol)
            
            # C. Mask ONLY the background (Standard Weight = 1.0)
            # This allows weights < 1.0 (0.9) and > 1.0 (2.0) to both appear.
            # We use isclose to handle floating point minor differences.
            background_value = 1.0
            masked_edge = np.ma.masked_where(np.isclose(edge_slice, background_value, atol=1e-3), edge_slice)
            
            # E. Plot
            im = axs[row_idx, 4].imshow(masked_edge.T, cmap='hot_r', alpha=0.6, vmin=0.0, vmax=1.0)
            
            axs[row_idx, 4].axis('off')
            axs[row_idx, 4].set_title(f"{view_name} - Soft Loss Map")

    return fig

# --- TRAINER CLASS ---

class nnUnet3DCustomTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # --- EXPERIMENT FLAGS ---
        self.EXP_MAG_ONLY_DUPLICATE = False 
        self.EXP_MAG_ONE_CHANNEL = False 
        self.EXP_OTSU = False
        self.EXP_MAG_PREPRO = False
        self.EXP_PHASE_PREPRO = False
        self.EXP_SPATIAL_AUGMENTATION_1 = False
        self.EXP_SPATIAL_AUGMENTATION_2 = False
        self.EXP_SPATIAL_AUGMENTATION_3 = False
        self.EXP_SPATIAL_AUGMENTATION_4 = False
        self.EXP_SPATIAL_AUGMENTATION_5 = False
        self.EXP_SPATIAL_AUGMENTATION_6 = False
        self.EXP_SPATIAL_AUGMENTATION_7 = False 
        self.EXP_SPATIAL_AUGMENTATION_8 = True
        self.EXP_SOFT_EDGE_LOSS_1 = False
        self.EXP_SOFT_EDGE_LOSS_2 = False
        self.EXP_SOFT_EDGE_LOSS_3 = False
        ## For EXP_SGD_OPTIMIZER please comment/uncomment as needed in the configure_optimizers function (done like this for simplicity)
        self.EXP_SGD_OPTIMIZER = False

        active_exps = [self.EXP_MAG_ONLY_DUPLICATE, self.EXP_MAG_ONE_CHANNEL, self.EXP_OTSU, self.EXP_MAG_PREPRO, self.EXP_PHASE_PREPRO, 
                       self.EXP_SPATIAL_AUGMENTATION_1, self.EXP_SPATIAL_AUGMENTATION_2, self.EXP_SPATIAL_AUGMENTATION_3, self.EXP_SPATIAL_AUGMENTATION_4,
                       self.EXP_SPATIAL_AUGMENTATION_5, self.EXP_SPATIAL_AUGMENTATION_6, self.EXP_SPATIAL_AUGMENTATION_7, self.EXP_SPATIAL_AUGMENTATION_8, 
                       self.EXP_SOFT_EDGE_LOSS_1, self.EXP_SOFT_EDGE_LOSS_2, self.EXP_SOFT_EDGE_LOSS_3]
        
        if sum(active_exps) > 1: raise RuntimeError("Multiple experiment flags set to True.")

        self.configuration = configuration
        self.plans = plans
        self.wandb_project = "MagPhase_MRI_Seg"
        self.wandb_run_name = f"Fold{fold}_{configuration}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        if self.EXP_SPATIAL_AUGMENTATION_1: self.wandb_run_name += "_spatial_aug_1_3d"
        if self.EXP_SPATIAL_AUGMENTATION_2: self.wandb_run_name += "_spatial_aug_2_3d"
        if self.EXP_SPATIAL_AUGMENTATION_3: self.wandb_run_name += "_spatial_aug_3_3d"
        if self.EXP_SPATIAL_AUGMENTATION_4: self.wandb_run_name += "_spatial_aug_4_3d"
        if self.EXP_SPATIAL_AUGMENTATION_5: self.wandb_run_name += "_spatial_aug_5_3d"
        if self.EXP_SPATIAL_AUGMENTATION_6: self.wandb_run_name += "_spatial_aug_6_3d"
        if self.EXP_SPATIAL_AUGMENTATION_7: self.wandb_run_name += "_spatial_aug_7_3d"
        if self.EXP_SPATIAL_AUGMENTATION_8: self.wandb_run_name += "_spatial_aug_8_3d"
        if self.EXP_SOFT_EDGE_LOSS_1: self.wandb_run_name += "_soft_edge_1"
        if self.EXP_SOFT_EDGE_LOSS_2: self.wandb_run_name += "_soft_edge_2"
        if self.EXP_SOFT_EDGE_LOSS_3: self.wandb_run_name += "_soft_edge_3"
        if self.EXP_MAG_ONLY_DUPLICATE: self.wandb_run_name += "_mag_only_duplicate"
        if self.EXP_MAG_ONE_CHANNEL: self.wandb_run_name += "_mag_one_channel"
        if self.EXP_MAG_PREPRO: self.wandb_run_name += "_mag_prepro"
        if self.EXP_PHASE_PREPRO: self.wandb_run_name += "_phase_prepro"
        if self.EXP_OTSU: self.wandb_run_name += "_otsu"
        if self.EXP_SGD_OPTIMIZER: self.wandb_run_name += "_sgd_opt"
        
        if self.EXP_PHASE_PREPRO or self.EXP_MAG_PREPRO: self.EXP_OTSU = True
           
        self.plot_next_val_sample = False
        self.plot_next_train_sample = False
        self.num_epochs = 200
        self.gpu_augmentation = None 
        self.initial_lr = 0.001
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.96)
        # lr_scheduler = CosineAnnealingLR(
            # optimizer, T_max=self.num_epochs, eta_min=0.00001, last_epoch=-1)
        return optimizer, lr_scheduler 
    
    def on_train_start(self):
        # 1. Initialize Standard Components (Network, Optimizer, etc.)
        super().on_train_start()
        
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        if self.local_rank == 0:
            config = {
                "fold": self.fold,
                "configuration": self.configuration_name,
                "batch_size": self.configuration_manager.batch_size,
                "EXP_SPATIAL_AUGMENTATION_1": self.EXP_SPATIAL_AUGMENTATION_1,
                "EXP_SPATIAL_AUGMENTATION_2": self.EXP_SPATIAL_AUGMENTATION_2,
                "EXP_SPATIAL_AUGMENTATION_3": self.EXP_SPATIAL_AUGMENTATION_3,
                "EXP_SPATIAL_AUGMENTATION_4": self.EXP_SPATIAL_AUGMENTATION_4,
                "EXP_SPATIAL_AUGMENTATION_5": self.EXP_SPATIAL_AUGMENTATION_5,
                "EXP_SPATIAL_AUGMENTATION_6": self.EXP_SPATIAL_AUGMENTATION_6,
                "EXP_SPATIAL_AUGMENTATION_7": self.EXP_SPATIAL_AUGMENTATION_7,
                "EXP_SPATIAL_AUGMENTATION_8": self.EXP_SPATIAL_AUGMENTATION_8,
                "EXP_SOFT_EDGE_LOSS_1": self.EXP_SOFT_EDGE_LOSS_1,
                "EXP_SOFT_EDGE_LOSS_2": self.EXP_SOFT_EDGE_LOSS_2,
                "EXP_SOFT_EDGE_LOSS_3": self.EXP_SOFT_EDGE_LOSS_3,
                "EXP_MAG_ONLY_DUPLICATE": self.EXP_MAG_ONLY_DUPLICATE,
                "EXP_MAG_ONE_CHANNEL": self.EXP_MAG_ONE_CHANNEL,
                "EXP_MAG_PREPRO": self.EXP_MAG_PREPRO,
                "EXP_PHASE_PREPRO": self.EXP_PHASE_PREPRO,
                "EXP_OTSU": self.EXP_OTSU,
                "EXP_SGD_OPTIMIZER": self.EXP_SGD_OPTIMIZER,
            }
            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=config)
        
        # 2. Set up Augmentations
        if self.EXP_SPATIAL_AUGMENTATION_1:
            self.gpu_augmentation = GPU3DSpatialAugmentation(
                patch_size=self.configuration_manager.patch_size,
                rot_range_deg=(30, 30, 30),
                scale_range=(0.7, 1.4),
                p_affine=0.3,
                p_flip=0.5
            ).to(self.device)
        
        if self.EXP_SPATIAL_AUGMENTATION_2:
            self.gpu_augmentation = GPU3DSpatialAugmentation(
                patch_size=self.configuration_manager.patch_size,
                rot_range_deg=(180, 180, 180),
                scale_range=(0.6, 1.5),
                p_affine=0.4,
                p_flip=0.5
            ).to(self.device)

        if self.EXP_SPATIAL_AUGMENTATION_3:
            self.gpu_augmentation = GPU3DSpatialAugmentation(
                patch_size=self.configuration_manager.patch_size,
                rot_range_deg=(90, 90, 90),
                scale_range=(0.8, 1.2),
                p_affine=0.3,
                p_flip=0.5
            ).to(self.device)
            
        if self.EXP_SPATIAL_AUGMENTATION_4:
            self.gpu_augmentation = GPU3DSpatialAugmentation(
                patch_size=self.configuration_manager.patch_size,
                rot_range_deg=(0, 180, 0),
                scale_range=(0.8, 1.2),
                p_affine=0.3,
                p_flip=0.5
            ).to(self.device)
        
        if self.EXP_SPATIAL_AUGMENTATION_5:
            self.gpu_augmentation = GPU3DSpatialAugmentation(
                patch_size=self.configuration_manager.patch_size,
                rot_range_deg=(0, 180, 0),
                scale_range=(0.4, 1.8),
                p_affine=0.7,
                p_flip=0.5
            ).to(self.device)
            
        if self.EXP_SPATIAL_AUGMENTATION_6:
            self.gpu_augmentation = GPU3DSpatialAugmentation(
                patch_size=self.configuration_manager.patch_size,
                rot_range_deg=(0, 90, 0),
                scale_range=(0.7, 1.7),
                trans_range=(0.45, 0.45, 0.45),
                shear_range_deg=(0, 35, 0, 0, 35, 0), 
                persp_factor=0.35, 
                keep_y_parallel=True,
                p_affine=0.95,
                p_flip=0.5
            ).to(self.device)

        if self.EXP_SPATIAL_AUGMENTATION_7:
            self.gpu_augmentation = GPU3DSpatialAugmentation(
                patch_size=self.configuration_manager.patch_size,
                rot_range_deg=(0, 180, 0),
                scale_range=(0.3, 2.0),
                trans_range=(0.45, 0.45, 0.45),
                shear_range_deg=(0, 55, 0, 0, 55, 0), 
                persp_factor=0.55, 
                keep_y_parallel=True,
                p_affine=0.95,
                p_flip=0.5
            ).to(self.device)

        if self.EXP_SPATIAL_AUGMENTATION_8:
            self.gpu_augmentation = GPU3DSpatialAugmentation(
                patch_size=self.configuration_manager.patch_size,
                rot_range_deg=(0, 180, 0),
                scale_range=(0.1, 3.0),
                trans_range=(0.8, 0.8, 0.8),
                shear_range_deg=(0, 85, 0, 0, 85, 0),
                persp_factor=0.85, 
                keep_y_parallel=True,
                p_affine=0.95,
                p_flip=0.5
            ).to(self.device)

        # 3. Override Loss with Soft Edge Loss if Enabled
        if self.EXP_SOFT_EDGE_LOSS_1:
            self.print_to_log_file("--> ACTIVATING CUSTOM 3D SOFT EDGE LOSS 1")
            self.edge_params = {
                1: {'edge_weight': 0.9, 'kernel_size': 7},
                2: {'edge_weight': 0.9, 'kernel_size': 3},  
                3: {'edge_weight': 0.6, 'kernel_size': 5},
                4: {'edge_weight': 0.4, 'kernel_size': 7}
            }
            
        if self.EXP_SOFT_EDGE_LOSS_2:
            self.print_to_log_file("--> ACTIVATING CUSTOM 3D SOFT EDGE LOSS 2")
            self.edge_params = {
                1: {'edge_weight': 0.9, 'kernel_size': 7},
                2: {'edge_weight': 0.9, 'kernel_size': 3},  
                3: {'edge_weight': 0.6, 'kernel_size': 3},
                4: {'edge_weight': 0.4, 'kernel_size': 3}
            }
            
        if self.EXP_SOFT_EDGE_LOSS_3:
            self.print_to_log_file("--> ACTIVATING CUSTOM 3D SOFT EDGE LOSS 3")
            self.edge_params = {
                1: {'edge_weight': 0.7, 'kernel_size': 5},
                2: {'edge_weight': 0.6, 'kernel_size': 3},  
                3: {'edge_weight': 0.2, 'kernel_size': 3},
                4: {'edge_weight': 0.2, 'kernel_size': 3}
            }
            
            # Compute deep supervision weights (replicating standard behavior)
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights = weights / weights.sum()

            # Instantiate Custom Loss
            loss = DC_and_CE_with_Edge_Loss(
                soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice, 
                                  'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                ce_kwargs={},
                weight_ce=6.0,
                weight_dice=1.0,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
                edge_params=self.edge_params,
                blur_sigma=1.0 # Sigma for 3D Gaussian Blur
            )
            
            # Wrap in Deep Supervision
            self.loss = DeepSupervisionWrapper(loss, weights)
            self.loss.to(self.device)

    def on_train_end(self):
        super().on_train_end()
        if self.local_rank == 0: wandb.finish()

    def on_epoch_start(self):
        super().on_epoch_start()
        self.plot_next_train_sample = True
        self.plot_next_val_sample = True

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.local_rank == 0:
            wandb.log({"Hyperparameters/Learning_Rate": self.optimizer.param_groups[0]['lr']}, step=self.current_epoch)
            
            if len(self.logger.my_fantastic_logging['train_losses']) > 0:
                wandb.log({"Train/Total_Loss": self.logger.my_fantastic_logging['train_losses'][-1]}, step=self.current_epoch)
            
            if len(self.logger.my_fantastic_logging['val_losses']) > 0:
                wandb.log({"Val/Total_Loss": self.logger.my_fantastic_logging['val_losses'][-1]}, step=self.current_epoch)

            # Log Dice scores
            if 'dice_per_class_or_region' in self.logger.my_fantastic_logging:
                 dice_scores = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
                 
                 if self.label_manager.has_regions:
                     names = self.label_manager.region_names
                 else:
                     labels_map = self.dataset_json.get('labels', {})
                     id_to_name = {v: k for k, v in labels_map.items() if isinstance(v, int)}
                     names = [id_to_name.get(lbl, f"Class_{lbl}") for lbl in self.label_manager.foreground_labels]

                 for name, score in zip(names, dice_scores):
                     wandb.log({f"Val_Dice/{name}": score}, step=self.current_epoch)
                 
                 if 'mean_fg_dice' in self.logger.my_fantastic_logging:
                     wandb.log({"Val/Mean_Foreground_Dice": self.logger.my_fantastic_logging['mean_fg_dice'][-1]}, step=self.current_epoch)

    def _downsample_target_on_gpu(self, target: torch.Tensor) -> list:
        ds_scales = self._get_deep_supervision_scales()
        downsampled_targets = [target]
        for scale in ds_scales[1:]:
            new_shape = [int(target.shape[i+2] * scale[i]) for i in range(3)]
            ds_target = F.interpolate(target.float(), size=new_shape, mode='nearest')
            downsampled_targets.append(ds_target.long())
        return downsampled_targets

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = target[0].to(self.device, non_blocking=True)
        else:
            target = target.to(self.device, non_blocking=True)

        if self.EXP_MAG_ONLY_DUPLICATE: data[:, 1] = data[:, 0]
            
        if (self.EXP_SPATIAL_AUGMENTATION_1 or self.EXP_SPATIAL_AUGMENTATION_2 or self.EXP_SPATIAL_AUGMENTATION_3  
            or self.EXP_SPATIAL_AUGMENTATION_4 or self.EXP_SPATIAL_AUGMENTATION_5 or self.EXP_SPATIAL_AUGMENTATION_6 
            or self.EXP_SPATIAL_AUGMENTATION_7  or self.EXP_SPATIAL_AUGMENTATION_8) and self.gpu_augmentation is not None:
            data, target = self.gpu_augmentation(data, target)

        if self.EXP_OTSU:
             data = _preprocess_data_gpu_3d(data, mag_prepro=self.EXP_MAG_PREPRO, phase_prepro=self.EXP_PHASE_PREPRO)

        if self.enable_deep_supervision:
            target_list = self._downsample_target_on_gpu(target)
        else:
            target_list = target

        self.optimizer.zero_grad()
        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            l = self.loss(output, target_list)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # --- VISUALIZATION BLOCK ---
        if self.local_rank == 0 and self.plot_next_train_sample and \
           (self.current_epoch % 10 == 0 or self.current_epoch == self.num_epochs - 1 or self.current_epoch <= 6):
            self.plot_next_train_sample = False
            try:
                raw_input = data.detach().cpu().float().numpy()
                gt_plot = target_list[0] if isinstance(target_list, list) else target_list
                gt_tensor = gt_plot.detach().cpu().numpy()[0]
                if gt_tensor.ndim == 4: gt_tensor = gt_tensor[0]
                
                output_final = output[0] if isinstance(output, (list, tuple)) else output
                pred_probs = torch.sigmoid(output_final[0])
                pred_vol = torch.argmax(pred_probs, dim=0).detach().cpu().numpy()
                
                mag_vol = raw_input[0, 0]
                phase_vol = raw_input[0, 1] if raw_input.shape[1] > 1 else np.zeros_like(mag_vol)
                
                # Extract Edge Map if available (SOFT LOSS)
                edge_vol = None
                if hasattr(self.loss, 'last_weight_map') and self.loss.last_weight_map is not None:
                    # Map is (Batch, 1, D, H, W). Take Batch 0, Channel 0.
                    w_map = self.loss.last_weight_map
                    if w_map.ndim == 5:
                        edge_vol = w_map[0, 0].detach().cpu().numpy()
                    else:
                        edge_vol = w_map[0].detach().cpu().numpy()

                fig = plot_3d_snapshot(
                    mag_vol, phase_vol, gt_tensor, pred_vol, 
                    num_classes=len(self.label_manager.foreground_labels), 
                    epoch=self.current_epoch, 
                    title_prefix="Train (Augmented)",
                    edge_vol=edge_vol # Pass the map
                )
                wandb.log({"Visuals/Train_Snapshot": wandb.Image(fig)}, step=self.current_epoch)
                plt.close(fig)
            except Exception as e: print(f"Train plot failed: {e}")

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list): target = [t.to(self.device, non_blocking=True) for t in target]
        else: target = target.to(self.device, non_blocking=True)

        if self.EXP_MAG_ONLY_DUPLICATE: data[:, 1] = data[:, 0]
        if self.EXP_OTSU:
             data = _preprocess_data_gpu_3d(data, mag_prepro=self.EXP_MAG_PREPRO, phase_prepro=self.EXP_PHASE_PREPRO)
        
        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output_final, target_final = output[0], target[0]
        else:
            output_final, target_final = output, target

        if self.local_rank == 0 and self.plot_next_val_sample and \
           (self.current_epoch % 10 == 0 or self.current_epoch == self.num_epochs - 1 or self.current_epoch == 1 or self.current_epoch == 2):
            self.plot_next_val_sample = False
            try:
                raw_input = data.cpu().float().numpy()
                gt_tensor = target_final.cpu().numpy()[0]
                if gt_tensor.ndim == 4: gt_tensor = gt_tensor[0]
                pred_probs = torch.sigmoid(output_final[0])
                pred_vol = torch.argmax(pred_probs, dim=0).cpu().numpy()
                
                mag_vol = raw_input[0, 0]
                phase_vol = raw_input[0, 1] if raw_input.shape[1] > 1 else np.zeros_like(mag_vol)
                
                # Try to extract map for validation too (might be None depending on loss state)
                edge_vol = None
                if hasattr(self.loss, 'last_weight_map') and self.loss.last_weight_map is not None:
                     w_map = self.loss.last_weight_map
                     if w_map.ndim == 5:
                         edge_vol = w_map[0, 0].detach().cpu().numpy()
                     else:
                         edge_vol = w_map[0].detach().cpu().numpy()

                fig = plot_3d_snapshot(
                    mag_vol, phase_vol, gt_tensor, pred_vol, 
                    num_classes=len(self.label_manager.foreground_labels), 
                    epoch=self.current_epoch, 
                    title_prefix="Validation",
                    edge_vol=edge_vol
                )
                wandb.log({"Visuals/Val_Snapshot": wandb.Image(fig)}, step=self.current_epoch)
                plt.close(fig)
            except Exception as e: print(f"Val plot failed: {e}")

        axes = [0] + list(range(2, output_final.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_final) > 0.5).long()
        else:
            output_seg = output_final.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_final.shape, device=output_final.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_final != self.label_manager.ignore_label).float()
                target_final[target_final == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target_final[:, -1:]
                target_final = target_final[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_final, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}