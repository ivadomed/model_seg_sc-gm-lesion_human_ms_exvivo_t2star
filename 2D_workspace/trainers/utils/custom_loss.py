# custom_loss.py (2D Version with Visualization)

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Type

# nnU-Net specific imports
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, SoftDiceLoss

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"All args must be either tuple or list, got {[type(i) for i in args]}"
        all_inputs = list(zip(*args))
        total_loss = 0.0
        if self.weight_factors[0] != 0:
            primary_loss = self.loss(*all_inputs[0])
            total_loss += self.weight_factors[0] * primary_loss
        for i in range(1, len(all_inputs)):
            if self.weight_factors[i] != 0:
                loss_i = self.loss(*all_inputs[i])
                total_loss += self.weight_factors[i] * loss_i
        with torch.no_grad():
            self.loss(*all_inputs[0])
        return total_loss

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.loss, name)

class DC_and_CE_with_Edge_Loss(nn.Module):
    def __init__(self,
                 soft_dice_kwargs: Dict,
                 ce_kwargs: Dict,
                 weight_ce: float = 1.0,
                 weight_dice: float = 1.0,
                 ignore_label: int = None,
                 dice_class: Type[Union[SoftDiceLoss, MemoryEfficientSoftDiceLoss]] = MemoryEfficientSoftDiceLoss,
                 edge_params: Dict = None,
                 blur_sigma: float = 1.0):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label
        self.edge_params = edge_params if edge_params else {}
        self.dice_loss = 0.0
        self.ce_loss = 0.0
        self.last_weight_map = None 

        if self.ignore_label is not None:
            ce_kwargs['ignore_index'] = self.ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def _create_edge_mask_and_vis(self, target_one_hot: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, float], torch.Tensor]:
        """
        Handles input shape: [Class, Batch, 1, D, H, W] -> (5, 2, 1, 192, 64, 208)
        max_pool3d requires: [Batch, Channel, D, H, W]
        """
        shape = target_one_hot.shape 
        
        # Dim 0 is Class, Dim 1 is Batch
        # shape[0]: Class (5)
        # shape[1]: Batch (2)
        # shape[2]: 1 (Dummy)
        # shape[3,4,5]: D, H, W
        
        # We assume 3D if 6 dims are present
        is_3d = (len(shape) == 6)
        
        if is_3d:
            pool_op = F.max_pool3d
            mask_shape = shape # Keep [Class, Batch, 1, D, H, W]
            
            # Visualization map should be [Batch, 1, D, H, W]
            vis_shape = (shape[1], 1, shape[3], shape[4], shape[5]) 
        else:
            # Fallback for 2D if needed (assuming standard formatting [B, C, H, W])
            pool_op = F.max_pool2d
            vis_shape = (shape[0], 1, shape[3], shape[4]) 
            mask_shape = shape

        # 1. Logic Mask (Matches input [Class, Batch, ...])
        edge_mask_map = torch.zeros(mask_shape, device=target_one_hot.device, dtype=torch.bool)
        
        # 2. Visualization Map (Matches Batch [Batch, 1, ...])
        vis_map = torch.ones(vis_shape, device=target_one_hot.device, dtype=torch.float32)
        
        class_soft_values = {}

        for class_idx, params in self.edge_params.items():
            k = params.get('kernel_size', 3)
            w_soft = params.get('edge_weight', 1.0)
            class_soft_values[class_idx] = w_soft
            
            pad = (k - 1) // 2 if k > 1 else 0
            
            if is_3d:
                # ---------------------------------------------------------
                # STEP 1: Extract Class (Slice Dim 0)
                # Input: [Class, Batch, 1, D, H, W]
                # Slice: [1, Batch, 1, D, H, W]
                # ---------------------------------------------------------
                
                class_mask_sliced = target_one_hot[class_idx:class_idx+1, ...] 
                # ---------------------------------------------------------
                # STEP 2: Prepare for max_pool3d (Need [Batch, 1, D, H, W])
                # Permute: [Batch, 1, 1, D, H, W]
                # Squeeze(2): [Batch, 1, D, H, W]
                # ---------------------------------------------------------
                mask_for_pool = class_mask_sliced.permute(1, 0, 2, 3, 4, 5).squeeze(2)
                
                # ---------------------------------------------------------
                # STEP 3: Apply Morphological Gradient
                # ---------------------------------------------------------
                dilated = pool_op(mask_for_pool.float(), kernel_size=k, stride=1, padding=pad)
                eroded = -pool_op(-mask_for_pool.float(), kernel_size=k, stride=1, padding=pad)
                
                # edge_region_bool shape: [Batch, 1, D, H, W]
                edge_region_bool = (dilated - eroded) > 0.5 

                # ---------------------------------------------------------
                # STEP 4: Assign back to [Class, Batch, 1, D, H, W]
                # Unsqueeze(2) -> [Batch, 1, 1, D, H, W]
                # Permute back -> [1, Batch, 1, D, H, W]
                # ---------------------------------------------------------
                edge_region_bool_orig_shape = edge_region_bool.unsqueeze(2).permute(1, 0, 2, 3, 4, 5)
                
                edge_mask_map[class_idx:class_idx+1, ...] = edge_region_bool_orig_shape
                
                # Update Vis Map (shape is [Batch, 1, D, H, W], so we use the pooled shape)
                

            else:
                squeezed_target = target_one_hot
                if len(shape) == 5:
                    squeezed_target = target_one_hot.squeeze(2)
                
                class_mask_sliced = squeezed_target[class_idx:class_idx+1, ...]
                
                dilated = pool_op(class_mask_sliced.float(), kernel_size=k, stride=1, padding=pad)
                eroded = -pool_op(-class_mask_sliced.float(), kernel_size=k, stride=1, padding=pad)
                
                edge_region_bool = (dilated - eroded) > 0.5 
                
                edge_mask_map_squeezed = edge_mask_map.squeeze(2)

                edge_mask_map_squeezed[class_idx:class_idx+1, ...] = edge_region_bool
                                    
            vis_map = torch.where(edge_region_bool, torch.tensor(w_soft, device=vis_map.device), vis_map)                
        
        if len(shape) == 5 and not is_3d:
            vis_map = vis_map.unsqueeze(2)
        
        return edge_mask_map, class_soft_values, vis_map
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        if isinstance(target, list):
            target = torch.stack(target, dim=0)
        
        mask = None
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label not implemented for one hot'
            mask = (target != self.ignore_label).bool()
            target_clean = torch.where(mask, target, torch.tensor(0, device=target.device, dtype=target.dtype))
        else:
            target_clean = target

        num_classes = net_output.shape[1]
        target_one_hot = convert_labelmap_to_one_hot(target_clean, list(range(num_classes)))
        
        # Check if we need to permute target_one_hot to match your [Class, Batch] expectation
        # Standard nnU-Net convert_labelmap_to_one_hot returns [Batch, Class, ...]
        # If your logs show [5, 2, ...], it might have been permuted somewhere or 
        # convert_labelmap_to_one_hot works differently in your version.
        # IF target_one_hot is [Batch, Class, ...] (e.g. [2, 5, ...]) but you need [Class, Batch, ...]
        if target_one_hot.shape[0] != num_classes and target_one_hot.shape[1] == num_classes:
             # Permute to [Class, Batch, ...] to match your logic
             target_one_hot = target_one_hot.permute(1, 0, *range(2, len(target_one_hot.shape)))

        target_soft = target_one_hot.float()

        # --- Generate Edges ---
        if self.edge_params:
            edge_mask_map, class_soft_values, vis_map = self._create_edge_mask_and_vis(target_one_hot)
            self.last_weight_map = vis_map.detach()
            
            for class_idx, alpha in class_soft_values.items():
                class_edges = edge_mask_map[class_idx:class_idx+1, ...] # Slice Dim 0
                
                # Apply Soft Labels
                # Shapes should now align: [1, Batch, 1, D, H, W]
                mask_fg = (target_one_hot[class_idx:class_idx+1, ...] == 1)
                mask_bg = (target_one_hot[class_idx:class_idx+1, ...] == 0)
                
                target_soft[class_idx:class_idx+1, ...][class_edges & mask_fg] = alpha
                target_soft[class_idx:class_idx+1, ...][class_edges & mask_bg] = 1.0 - alpha
        else:
            self.last_weight_map = None

        # --- Dice Loss ---
        # Net output is typically [Batch, Class, D, H, W]
        # Target soft is currently [Class, Batch, 1, D, H, W] (based on your 6D shape)
        
        # We need to align target_soft to net_output for Dice Loss
        # 1. Permute back to [Batch, Class, 1, D, H, W]
        if target_soft.dim() == 6: 
            target_soft_for_loss = target_soft.permute(1, 0, 2, 3, 4, 5)
        else : 
            target_soft_for_loss = target_soft.permute(1, 0, 2, 3, 4)
        
        # 2. Squeeze the dummy dim 2 -> [Batch, Class, D, H, W]
        if target_soft_for_loss.shape[2] == 1:
            target_soft_for_loss = target_soft_for_loss.squeeze(2)
        
        dc_loss = self.dc(net_output, target_soft_for_loss, loss_mask=mask) if self.weight_dice != 0 else 0
        self.dice_loss = dc_loss
        
        # CE Loss
        # target[:, 0] is usually [Batch, D, H, W] (Integer labels)
        ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 else 0
        self.ce_loss = ce_loss
        
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result