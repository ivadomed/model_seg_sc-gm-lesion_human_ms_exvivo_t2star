"""GPU spatial augmentation for offline distillation.

Subclasses the base `GPU3DSpatialAugmentation` and applies the SAME sampled flip + affine grid to
data, segmentation target AND the teacher logit map in one call, so all three stay voxel-aligned.
The teacher map is continuous, so it is resampled with `bilinear` (like data), not `nearest`
(like the integer segmentation).

Installed flat into the venv `variants/` dir (see scripts/02_train/01_install_trainer.sh).
"""
import random

import torch
import torch.nn.functional as F

from .augmentation_3D import GPU3DSpatialAugmentation


class GPU3DSpatialAugmentationKD(GPU3DSpatialAugmentation):
    def forward(self, data, target, teacher):
        B, C, D, H, W = data.shape
        device = data.device

        if torch.rand(1) < self.p_flip:
            for dim in (2, 4):  # flip Depth or Width, preserve Y (slice axis)
                if random.random() < 0.5:
                    data = torch.flip(data, [dim])
                    target = torch.flip(target, [dim])
                    teacher = torch.flip(teacher, [dim])

        if random.random() < self.p_affine:
            fwd_matrix = self._get_transform_matrix(B, device)
            try:
                inv_matrix = torch.linalg.inv(fwd_matrix)
            except RuntimeError:
                inv_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)

            grid_flat = self._create_perfect_grid(data.shape, device)
            grid_transformed = torch.bmm(inv_matrix, grid_flat)

            w = torch.clamp(grid_transformed[:, 3, :], min=1e-4)
            x = grid_transformed[:, 0, :] / w
            z = grid_transformed[:, 2, :] / w
            y = grid_transformed[:, 1, :] if self.keep_y_parallel else grid_transformed[:, 1, :] / w
            grid = torch.stack([x, y, z], dim=2).reshape(B, D, H, W, 3)

            data = F.grid_sample(data, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            target = F.grid_sample(target.float(), grid, mode='nearest', padding_mode='zeros', align_corners=False).long()
            teacher = F.grid_sample(teacher, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return data, target, teacher
