"""
nnUNetTrainerWandb trainer class for nnUNet that integrates with Weights & Biases (wandb).
This class extends the nnUNetTrainer class to log training and validation metrics, images, and
other relevant information to wandb for visualization and tracking.

This version is corrected to use the proper attributes and includes robust, multi-channel plotting
with a bitmask decoder for ground truth visualization.
"""
from datetime import datetime
import matplotlib.pyplot as plt 
import numpy as np
from torch.cuda.amp import autocast
from contextlib import nullcontext as dummy_context
import os
import torch
import wandb
import copy
import tqdm

from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from skimage.restoration import unwrap_phase
from skimage.exposure import equalize_adapthist as clahe, rescale_intensity

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import RicianNoiseTransform

import torch
from torch import nn
from nnunetv2.training.loss.dice import SoftDiceLoss, get_tp_fp_fn_tn

class WeightedSoftDiceLoss(SoftDiceLoss):
    """
    A custom SoftDiceLoss that applies channel-specific weights.
    This is crucial for tasks with significant class imbalance, especially when
    dealing with region-based segmentation where intersections of labels are rare.

    Args:
        weight (torch.Tensor): A tensor of weights for each foreground channel.
        apply_nonlin, batch_dice, do_bg, smooth: Same as parent SoftDiceLoss.
    """
    def __init__(self, weight: torch.Tensor, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-5):
        super().__init__(apply_nonlin=apply_nonlin, batch_dice=batch_dice, do_bg=do_bg, smooth=smooth)
        self.weight = weight

    def forward(self, x, y, loss_mask=None):
        if isinstance(x, (list, tuple)):
            x = x[0]
            
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)

        y_one_hot = y[0]
        axes = tuple(range(2, x.ndim))
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_one_hot, axes=axes, mask=loss_mask)

        numerator = 2 * tp
        denominator = 2 * tp + fp + fn + self.smooth
        
        dice_per_channel = numerator / denominator

        loss = torch.mean((1 - dice_per_channel) * self.weight)

        return loss
    
    
    def get_train_transform(self):
        """
        Defines and returns the aggressive data augmentation pipeline.
        Spatial transforms are applied to both channels.
        Intensity transforms are applied ONLY to the magnitude channel (0).
        """
        shared_transforms = [
            SpatialTransform(
                patch_size=self.configuration_manager.patch_size,
                patch_center_dist_from_border=None,
                do_elastic_deform=True, alpha=(0, 1200), sigma=(10, 15),
                do_rotation=True, angle_x=(-0.4, 0.4), angle_y=(-0.4, 0.4), angle_z=(-0.4, 0.4), # Aggressive rotation
                do_scale=True, scale=(0.6, 1.4), # Aggressive scaling
                border_mode_data='constant', border_cval_data=0,
                order_data=3, random_crop=True
            ),
            MirrorTransform(axes=(0, 1, 2)) # Random flipping on all axes
        ]

        # Define transforms that should ONLY be applied to the magnitude channel (channel 0)
        magnitude_only_transforms = [
            BrightnessMultiplicativeTransform(multiplier_range=(0.5, 2.0), per_channel=False), # Aggressive brightness
            GammaTransform(gamma_range=(0.5, 2.0), per_channel=False), # Aggressive gamma
            RicianNoiseTransform(noise_variance=(0, 0.1), per_channel=False) # Increased noise
        ]

        return Compose(shared_transforms)

def decode_bitmask_to_multi_channel(bitmask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Decodes a single-channel integer bitmask into a multi-channel binary mask.
    This is used to visualize the ground truth correctly.
    """
    h, w = bitmask.shape
    multi_channel_mask = np.zeros((num_classes, h, w), dtype=np.uint8)
    for i in range(num_classes):
        multi_channel_mask[i] = ((bitmask & (2**(i))) > 0).astype(np.uint8)
    return multi_channel_mask

def decode_bitmask_to_7_channels(bitmask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Decodes a single-channel integer bitmask into a 7-channel binary mask,
    where each channel corresponds to a unique foreground value (1 through num_classes).
    """
    h, w = bitmask.shape
    multi_channel_mask = np.zeros((num_classes, h, w), dtype=np.uint8)
    
    for i in range(1, num_classes + 1):
        channel_idx = i - 1
        multi_channel_mask[channel_idx][bitmask == i] = 1
        
    return multi_channel_mask


def plot_comparison_and_segmentation(
    orig_mag, orig_phase,
    proc_mag, proc_phase,
    gt_masks, pred_masks_full, pred_masks_simplified,
    aug_mag=None, aug_phase=None 
):
    colors = ['black', 'cyan', 'lime', 'red', 'yellow', 'magenta', 'orange', 'purple']
    fig, axs = plt.subplots(4, 3, figsize=(18, 24), gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
    fig.suptitle('Input Comparison and Segmentation Results', fontsize=20)

    axs[0, 0].imshow(orig_mag.T, cmap='gray')
    axs[0, 0].set_title("Original Magnitude")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(orig_phase.T, cmap='gray')
    axs[0, 1].set_title("Original Phase")
    axs[0, 1].axis('off')
    axs[0, 2].axis('off')

    if aug_mag is not None:
        axs[1, 0].imshow(aug_mag.T, cmap='gray')
        axs[1, 0].set_title("Augmented Magnitude")
    axs[1, 0].axis('off')

    if aug_phase is not None:
        axs[1, 1].imshow(aug_phase.T, cmap='gray')
        axs[1, 1].set_title("Augmented Phase")
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')

    axs[2, 0].imshow(proc_mag.T, cmap='gray')
    axs[2, 0].set_title("Preprocessed Magnitude")
    axs[2, 0].axis('off')

    axs[2, 1].imshow(proc_phase.T, cmap='gray')
    axs[2, 1].set_title("Preprocessed Phase")
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')

    segmentation_plots = [
        (gt_masks, "Ground Truth"),
        (pred_masks_full, "Prediction (Multi-Label)"),
        (pred_masks_simplified, "Prediction (Winner-Takes-All)")
    ]

    for i, (masks, title) in enumerate(segmentation_plots):
        ax = axs[3, i]
        ax.imshow(proc_mag.T, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

        if masks is not None and masks.ndim == 3:
            for class_idx in range(masks.shape[0]):
                adjustment = 1 if i == 0 else 0
                color = colors[(class_idx + adjustment) % len(colors)]
                if np.any(masks[class_idx]):
                    ax.contourf(masks[class_idx].T, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                    ax.contour(masks[class_idx].T, levels=[0.5], colors=[color], linewidths=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

class nnUNetTrainerWandb(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_segmentation_classes = len(self.label_manager.foreground_regions) if self.label_manager.has_regions else len(self.label_manager.foreground_labels)
        self.num_segmentation_classes += 1 ## because of the background class
        self.data_aug = self.get_train_transform()
    
    def get_train_transform(self):
        """
        Defines and returns the aggressive data augmentation pipeline.
        This method now lives in the Trainer class where it belongs.
        """
        patch_size = self.configuration_manager.patch_size
        patch_center_dist_from_border = [i // 2 for i in patch_size]
        # Define transforms that should be applied to BOTH magnitude and phase
        shared_transforms = [
            SpatialTransform(
                patch_size=self.configuration_manager.patch_size,
                patch_center_dist_from_border=patch_center_dist_from_border,
                do_elastic_deform=False, alpha=(0, 200), sigma=(10, 15),
                do_rotation=True, angle_x=(-0.4, 0.4), angle_y=(-0.4, 0.4), angle_z=(-0.4, 0.4),
                do_scale=True, scale=(0.6, 1.4),
                border_mode_data='constant', border_cval_data=0,
                order_data=3, order_seg=0,
                random_crop=True
            ),
            MirrorTransform(axes=(0, 1, 2))
        ]

        # Define transforms that should ONLY be applied to the magnitude channel (channel 0)
        magnitude_only_transforms = [
            BrightnessMultiplicativeTransform(multiplier_range=(0.5, 2.0), per_channel=False),
            GammaTransform(gamma_range=(0.5, 2.0), per_channel=False),
            RicianNoiseTransform(noise_variance=(0, 0.1))
        ]

        return Compose(shared_transforms)
    
    
    def _preprocess_data(self, data_batch: torch.Tensor) -> torch.Tensor:
        """
        Applies custom preprocessing.
        - Magnitude: CLAHE
        - Phase: linear contrast
        """
        batch_np = data_batch.cpu().numpy()
        processed_batch_np = np.zeros_like(batch_np)

        for i in range(batch_np.shape[0]):
            mag_channel = batch_np[i, 0]
            vmin, vmax = mag_channel.min(), mag_channel.max()
            if vmax > vmin:
                mag_normalized = (mag_channel - vmin) / (vmax - vmin)
            else:
                mag_normalized = np.zeros_like(mag_channel) 
            
            mag_clahe = clahe(mag_normalized, clip_limit=0.01)
            processed_batch_np[i, 0] = mag_clahe
            
            phase_channel = batch_np[i, 1]
            
            unwrapped_phase = unwrap_phase(phase_channel)
            
            p2, p98 = np.percentile(unwrapped_phase, (30, 75))
            phase_final = rescale_intensity(unwrapped_phase, in_range=(p2, p98), out_range=(0, 1))

            processed_batch_np[i, 1] = phase_final

        return torch.from_numpy(processed_batch_np).float()

    def on_train_start(self):
        """
        --- ADDED/MODIFIED ---
        Initializes the weighted loss function. This is the correct place to do it,
        ensuring the weight tensor is created on the final target device.
        """
        super().on_train_start() 
        self.num_epochs = 350
        self.initial_lr = 1e-3
        channel_weights = torch.tensor([
            1.0, # BG
            2.0, # SC only
            0.0, # GM only
            0.0, # Lesion only
            3.0, # SC + GM
            4.0, # SC + Lesion
            0.0, # GM + Lesion
            8.0  # SC + GM + Lesion
        ], device=self.device)

        # Use the WeightedSoftDiceLoss
        # self.loss = WeightedSoftDiceLoss(
        #     weight=channel_weights,
        #     apply_nonlin=torch.sigmoid,
        #     batch_dice=True,
        #     do_bg=False,
        #     smooth=1e-5
        # )

    def run_training(self):
        self.on_train_start()
        self.wandb_init()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
               train_outputs.append(self.train_step(next(self.dataloader_train), batch_id, epoch))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val), batch_id, epoch))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
        wandb.finish()  
        self.on_train_end()

    def wandb_init(self):
        project_name = f"'model_seg_sc-gm-lesion_human_ms_exvivo_t2star'-{self.plans_manager.dataset_name}"
        wandb.init(
            project=project_name,
            name=f"{self.configuration_name}_fold{self.fold}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            config={
                "plans_name": self.plans_manager.plans_name, "configuration": self.configuration_name, "fold": self.fold,
                "batch_size": self.batch_size, "initial_lr": self.initial_lr, "num_epochs": self.num_epochs,
                "plans": self.plans_manager.plans
            }
        )
        wandb.watch(self.network, log="all", log_freq=self.num_iterations_per_epoch)

        
    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        
        avg_train_loss = np.mean([i['loss'] for i in train_outputs])
        wandb.log({"Training/Loss": avg_train_loss}, step=self.current_epoch)
        
    def on_epoch_end(self):
        super().on_epoch_end()
        wandb.log({"Hyperparameters/Learning_Rate": self.optimizer.param_groups[0]['lr']}, step=self.current_epoch)

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)

        mean_fg_dice = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
        val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
        
        wandb_log_dict = {
            "Validation/Mean_Foreground_Dice": mean_fg_dice,
            "Validation/Loss": val_loss,
        }

        dice_per_class_or_region = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        if self.label_manager.has_regions:
            for i, name in enumerate(self.label_manager.region_names):
                wandb_log_dict[f'Validation_Dice/{name}'] = dice_per_class_or_region[i]
        else:
            for i, label_id in enumerate(self.label_manager.foreground_labels+[8]):
                wandb_log_dict[f'Validation_Dice/Class_{label_id-1}'] = dice_per_class_or_region[i]
        
        wandb.log(wandb_log_dict, step=self.current_epoch)

    def train_step(self, batch: dict, batch_id: int, epoch_id: int) -> dict:
        data, target = batch['data'], batch['target']
        
        original_data_for_logging = data.clone()
        
        data_np = data.cpu().numpy()
        target_np = target[0].cpu().numpy()
        augmented_batch = self.data_aug(**{'data': data_np, 'seg': target_np})
        augmented_data = torch.from_numpy(augmented_batch['data'])
        target = [torch.from_numpy(augmented_batch['seg'])]
        
        preprocessed_data = self._preprocess_data(augmented_data)
        preprocessed_data = preprocessed_data.to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in target]
        
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(preprocessed_data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            self.optimizer.step()
        
        if batch_id == 0 and (self.current_epoch % 1 == 0 or self.current_epoch == self.num_epochs - 1):
            with torch.no_grad():
                output_for_plotting = output[0] if self.enable_deep_supervision else output
                target_for_plotting = target[0] if self.enable_deep_supervision else target

                orig_mag = original_data_for_logging[0, 0].cpu().numpy()
                orig_phase = original_data_for_logging[0, 1].cpu().numpy()
                
                aug_mag = augmented_data[0, 0].cpu().numpy()
                aug_phase = augmented_data[0, 1].cpu().numpy()
                
                proc_mag = preprocessed_data[0, 0].cpu().numpy()
                proc_phase = preprocessed_data[0, 1].cpu().numpy()
                
                gt_integer_map = target_for_plotting[0].squeeze().cpu().numpy()
                train_gt_masks = decode_bitmask_to_7_channels(gt_integer_map, self.num_segmentation_classes)

                train_probs = torch.sigmoid(output_for_plotting[0]).detach().cpu().numpy()
                train_pred_masks_full = (train_probs > 0.5).astype(np.uint8)
                
                winner_indices = np.argmax(train_probs, axis=0)
                winner_probs = np.max(train_probs, axis=0)
                is_confident = winner_probs > 0.5
                train_pred_masks_simplified = np.zeros_like(train_probs, dtype=np.uint8)

                num_predicted_channels = train_pred_masks_simplified.shape[0]
                for i in range(num_predicted_channels):
                    is_winner = (winner_indices == i)
                    train_pred_masks_simplified[i][is_winner & is_confident] = 1
                
                fig = plot_comparison_and_segmentation(
                    orig_mag=orig_mag, orig_phase=orig_phase,
                    aug_mag=aug_mag, aug_phase=aug_phase,
                    proc_mag=proc_mag, proc_phase=proc_phase,
                    gt_masks=train_gt_masks, 
                    pred_masks_full=train_pred_masks_full, 
                    pred_masks_simplified=train_pred_masks_simplified
                )
                wandb.log({"Training/Segmentation_Comparison": wandb.Image(fig)}, step=self.current_epoch)
                plt.close(fig)

        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict, batch_id: int, epoch_id: int) -> dict:
        data, target = batch['data'], batch['target']

        original_data = data.clone()
        preprocessed_data = self._preprocess_data(data)

        preprocessed_data = preprocessed_data.to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in target]

        with torch.no_grad():
            output = self.network(preprocessed_data) 
            l = self.loss(output, target)

            if self.enable_deep_supervision:
                output_for_eval = output[0]
                target_for_eval = target[0]
            else:
                output_for_eval = output
                target_for_eval = target[0]

            if batch_id == 0 and (self.current_epoch % 1 == 0 or self.current_epoch == self.num_epochs - 1):
                orig_mag = original_data[0, 0].cpu().numpy()
                orig_phase = original_data[0, 1].cpu().numpy()
                proc_mag = preprocessed_data[0, 0].cpu().numpy()
                proc_phase = preprocessed_data[0, 1].cpu().numpy()
                
                gt_integer_map = target_for_eval[0].squeeze().cpu().numpy()
                val_gt_masks = decode_bitmask_to_7_channels(gt_integer_map, self.num_segmentation_classes)
                
                pred_probs = torch.sigmoid(output_for_eval[0]).cpu().numpy()
                val_pred_masks_full = (pred_probs > 0.5).astype(np.uint8)

                winner_indices = np.argmax(pred_probs, axis=0)
                winner_probs = np.max(pred_probs, axis=0)
                is_confident = winner_probs > 0.5
                val_pred_masks_simplified = np.zeros_like(pred_probs, dtype=np.uint8)
                for i in range(self.num_segmentation_classes):
                    is_winner = (winner_indices == i)
                    val_pred_masks_simplified[i][is_winner & is_confident] = 1

                fig = plot_comparison_and_segmentation(
                    orig_mag, orig_phase, proc_mag, proc_phase,
                    val_gt_masks, val_pred_masks_full, val_pred_masks_simplified
                )
                wandb.log({"Validation/Segmentation_Comparison": wandb.Image(fig)}, step=self.current_epoch)
                plt.close(fig)

            axes = [0] + list(range(2, output_for_eval.ndim))
            predicted_segmentation_onehot = (torch.sigmoid(output_for_eval) > 0.5).long()
            
            if self.label_manager.has_ignore_label:
                mask = 1 - target_for_eval[:, -1:]
                target_bis = target_for_eval[:, :-1]
            else:
                mask = None
                target_bis = target_for_eval

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_bis, axes=axes, mask=mask)
            
        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp.detach().cpu().numpy(), 
                'fp_hard': fp.detach().cpu().numpy(), 'fn_hard': fn.detach().cpu().numpy()}