import torch
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
from datetime import datetime
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.helpers import empty_cache, dummy_context

# --- HELPER FUNCTIONS ---

def decode_bitmask_to_multichannel(bitmask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Decodes a single-channel integer bitmask into a multi-channel binary mask.
    Adapted from user provided 'decode_bitmask_to_7_channels'.
    """
    # bitmask shape: (H, W) or (H, W)
    h, w = bitmask.shape
    multi_channel_mask = np.zeros((num_classes, h, w), dtype=np.uint8)
    
    # We assume class 0 is background, so we loop 1 to num_classes
    # The output channels correspond to indices 0..N-1
    for i in range(1, num_classes + 1):
        channel_idx = i - 1
        if channel_idx < num_classes:
            multi_channel_mask[channel_idx][bitmask == i] = 1
        
    return multi_channel_mask

def plot_3d_snapshot(mag_vol, phase_vol, gt_vol, pred_vol, num_classes, epoch):
    """
    Plots Axial, Coronal, and Sagittal views of the middle slice.
    """
    # Shapes are expected to be 3D: (Z, Y, X)
    shp = mag_vol.shape
    mid_z, mid_y, mid_x = shp[0] // 2, shp[1] // 2, shp[2] // 2
    
    # Define views: (Name, Slice_Selector_Function)
    views = [
        ('Axial', lambda x: x[mid_z, :, :]),
        ('Coronal', lambda x: x[:, mid_y, :]),
        ('Sagittal', lambda x: x[:, :, mid_x])
    ]
    
    # Layout: Rows=Views, Cols=Mag, Phase, GT, Pred
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.suptitle(f'Validation Snapshot - Epoch {epoch}', fontsize=16)
    
    # Colors for segmentation overlay (Cyan, Lime, Red, Yellow, Magenta...)
    colors = ['cyan', 'lime', 'red', 'yellow', 'magenta', 'orange', 'purple']

    for row_idx, (view_name, slicer) in enumerate(views):
        # 1. Magnitude
        axs[row_idx, 0].imshow(slicer(mag_vol).T, cmap='gray')
        axs[row_idx, 0].set_title(f"{view_name} - Mag")
        axs[row_idx, 0].axis('off')

        # 2. Phase
        axs[row_idx, 1].imshow(slicer(phase_vol).T, cmap='gray')
        axs[row_idx, 1].set_title(f"{view_name} - Phase")
        axs[row_idx, 1].axis('off')

        # 3. Ground Truth (Overlay)
        # Background
        axs[row_idx, 2].imshow(slicer(mag_vol).T, cmap='gray', alpha=0.6)
        
        # Get integer mask slice
        gt_slice = slicer(gt_vol)
        
        # Decode to channels for contour plotting
        gt_channels = decode_bitmask_to_multichannel(gt_slice, num_classes)
        
        for c in range(num_classes):
            if np.any(gt_channels[c]):
                color = colors[c % len(colors)]
                # Contour filled for visibility
                axs[row_idx, 2].contourf(gt_channels[c].T, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                axs[row_idx, 2].contour(gt_channels[c].T, levels=[0.5], colors=[color], linewidths=1)
        
        axs[row_idx, 2].set_title(f"{view_name} - GT")
        axs[row_idx, 2].axis('off')

        # 4. Prediction (Overlay)
        axs[row_idx, 3].imshow(slicer(mag_vol).T, cmap='gray', alpha=0.6)
        
        pred_slice = slicer(pred_vol)
        pred_channels = decode_bitmask_to_multichannel(pred_slice, num_classes)
        
        for c in range(num_classes):
            if np.any(pred_channels[c]):
                color = colors[c % len(colors)]
                axs[row_idx, 3].contourf(pred_channels[c].T, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                axs[row_idx, 3].contour(pred_channels[c].T, levels=[0.5], colors=[color], linewidths=1)

        axs[row_idx, 3].set_title(f"{view_name} - Pred")
        axs[row_idx, 3].axis('off')

    return fig

# --- TRAINER CLASS ---

class nnUNetTrainerMagPhase(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # --- EXPERIMENT CONFIGURATION ---
        self.ENABLE_MAG_MAG_EXPERIMENT = False 
        
        # Wandb Config
        self.wandb_project = "MagPhase_MRI_Seg"
        self.wandb_run_name = f"Fold{fold}_{configuration}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if self.ENABLE_MAG_MAG_EXPERIMENT:
            self.wandb_run_name += "_MagMagAblation"
            
        # Helper flag to plot only the first validation batch
        self.plot_next_val_sample = False

    def on_train_start(self):
        # Initialize Wandb (Only on Rank 0 to avoid duplicates in DDP)
        if self.local_rank == 0:
            config = {
                "fold": self.fold,
                "configuration": self.configuration_name,
                "batch_size": self.batch_size,
                "patch_size": self.configuration_manager.patch_size,
                "initial_lr": self.initial_lr,
                "experiment_mag_mag": self.ENABLE_MAG_MAG_EXPERIMENT,
                "dataset": self.dataset_json.get('name', 'Unknown')
            }
            
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config=config
            )
        super().on_train_start()

    def on_train_end(self):
        super().on_train_end()
        if self.local_rank == 0:
            wandb.finish()

    def on_epoch_end(self):
        super().on_epoch_end()
        
        if self.local_rank == 0:
            # 1. Log Learning Rate
            wandb.log({"Hyperparameters/Learning_Rate": self.optimizer.param_groups[0]['lr']}, step=self.current_epoch)
            
            # 2. Log Train Metrics
            if len(self.logger.my_fantastic_logging['train_losses']) > 0:
                wandb.log({
                    "Train/Total_Loss": self.logger.my_fantastic_logging['train_losses'][-1]
                }, step=self.current_epoch)
                
            # 3. Log Validation Metrics
            if len(self.logger.my_fantastic_logging['val_losses']) > 0:
                wandb.log({
                    "Val/Total_Loss": self.logger.my_fantastic_logging['val_losses'][-1],
                    "Val/Mean_Foreground_Dice": self.logger.my_fantastic_logging['mean_fg_dice'][-1]
                }, step=self.current_epoch)
                
                # 4. Log Class-wise Dice
                dice_per_class = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
                labels = self.dataset_json.get('labels', {})
                # Reverse label map: {1: "White Matter", 2: "Gray Matter"...}
                label_names = {v: k for k, v in labels.items() if isinstance(v, int) and v > 0}
                
                # dice_per_class is a list of scores for foreground classes in order
                for i, dice_score in enumerate(dice_per_class):
                    class_id = i + 1 # Assuming 0 is BG
                    class_name = label_names.get(class_id, f"Class_{class_id}")
                    wandb.log({f"Val_Dice/{class_name}": dice_score}, step=self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        # Intercept for Mag-Mag Experiment
        if self.ENABLE_MAG_MAG_EXPERIMENT:
            batch['data'][:, 1] = batch['data'][:, 0]
        return super().train_step(batch)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        # Reset flag to ensure we plot the first batch of this epoch
        self.plot_next_val_sample = True

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # --- EXPERIMENT LOGIC ---
        if self.ENABLE_MAG_MAG_EXPERIMENT:
            data[:, 1] = data[:, 0]
        
        # --- INFERENCE ---
        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output_final = output[0]
            target_final = target[0]
        else:
            output_final = output
            target_final = target[0]

        # --- WANDB PLOTTING (First Batch Only) ---
        # We plot every 10 epochs (and the last one) to save space/time
        if self.local_rank == 0 and self.plot_next_val_sample and \
           (self.current_epoch % 10 == 0 or self.current_epoch == self.num_epochs - 1):
            
            self.plot_next_val_sample = False # Disable for rest of epoch
            
            # Move to CPU for plotting
            # data was deleted, so we grab from batch dict which is on CPU in standard DataLoader
            # BUT nnUNet DataLoader usually yields GPU tensors if pinned. 
            # Safest is to use the 'batch' input which is the source dict.
            
            try:
                # 1. Get Image (Batch index 0) - Shape (C, Z, Y, X)
                # If data was moved to GPU in-place, we might need to move it back.
                # However, batch['data'] in the signature is the original reference.
                img_tensor = batch['data'][0].cpu().numpy() 
                mag_vol = img_tensor[0]
                phase_vol = img_tensor[1]
                
                # 2. Get GT - Shape (1, Z, Y, X) or (Z, Y, X)
                gt_tensor = target_final.cpu().numpy()[0]
                if gt_tensor.ndim == 4: gt_tensor = gt_tensor[0]
                
                # 3. Get Prediction - Shape (Classes, Z, Y, X) -> Argmax -> (Z, Y, X)
                # Apply Sigmoid/Softmax depending on training mode, but Argmax is standard for viz
                pred_probs = torch.sigmoid(output_final[0])
                pred_vol = torch.argmax(pred_probs, dim=0).cpu().numpy()
                
                # 4. Generate Plot
                num_classes = len(self.label_manager.foreground_labels)
                fig = plot_3d_snapshot(mag_vol, phase_vol, gt_tensor, pred_vol, num_classes, self.current_epoch)
                
                # 5. Log
                wandb.log({"Visuals/3D_Snapshot": wandb.Image(fig)}, step=self.current_epoch)
                plt.close(fig)
                
            except Exception as e:
                print(f"Warning: Wandb plotting failed: {e}")

        # --- METRICS (Standard nnU-Net) ---
        # This part ensures nnU-Net's internal logger gets what it needs
        axes = [0] + list(range(2, output_final.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output_final) > 0.5).long()
        else:
            output_seg = output_final.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output_final.shape, device=output_final.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_final != self.label_manager.ignore_label).float()
                target_final[target_final == self.label_manager.ignore_label] = 0
            else:
                if target_final.dtype == torch.bool:
                    mask = ~target_final[:, -1:]
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