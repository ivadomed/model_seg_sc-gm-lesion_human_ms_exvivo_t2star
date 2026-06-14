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