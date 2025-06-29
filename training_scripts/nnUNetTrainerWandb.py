
"""
nnUNetTrainerWandb trainer class for nnUNet that integrates with Weights & Biases (wandb).
This class extends the nnUNetTrainer class to log training and validation metrics, images, and
other relevant information to wandb for visualization and tracking.

Inspired from https://github.com/MIC-DKFZ/nnUNet/issues/2733#issuecomment-2744210810
"""
from datetime import datetime
import matplotlib.pyplot as plt 
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext as dummy_context
import os
import torch
import wandb
import copy
import tqdm

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss


class nnUNetTrainerWandb(nnUNetTrainer):

    def run_training(self):
        self.on_train_start()
        output_path = os.path.join("output_path", str(datetime.now().date()) +"_" +str(datetime.now().time()))
        os.makedirs(output_path, exist_ok=True)

        wandb.init(project=f'model_seg_sc-gm-lesion_human_ms_exvivo_t2star',  dir=output_path)

        for epoch in tqdm.tqdm(range(self.current_epoch, self.num_epochs)):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in tqdm.tqdm(range(self.num_iterations_per_epoch)): # What ? Why ? Why not the whole thing ?
               train_outputs.append(self.train_step(next(self.dataloader_train),batch_id,epoch))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in tqdm.tqdm(range(self.num_val_iterations_per_epoch)): # What ? Why ? Why not the whole thing ?
                    val_outputs.append(self.validation_step(next(self.dataloader_val),batch_id,epoch))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
        wandb.finish()  
        self.on_train_end()

    def train_step(self, batch: dict, batch_id: int, epoch_id: int) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

            if batch_id == 0: 
                train_image= data[0].detach().cpu().squeeze().float().numpy()
                # Deal with region-based training
                if target[0].shape[1] == 1:
                    # Not region-based, just squeeze the class dimension
                    train_gt = target[0].detach().cpu().squeeze().float().numpy()[0]
                    train_pred = np.argmax(output[0].detach().cpu().squeeze().numpy(), axis=1)[0]                
                else:
                    # Region-based, we need to sum all classes along the class dimension
                    train_gt = target[0].detach().cpu().squeeze().float().numpy()[0]
                    train_gt = np.squeeze(np.sum(train_gt, axis=0))  # Sum across classes to get a single mask and squeeze it
                    train_pred = output[0].detach().cpu().squeeze().float().numpy()[0]
                    train_pred = np.squeeze(np.sum(train_pred, axis=0))  # Sum across classes to get a single mask and squeeze it
                fig = plot_single_slice(combined=train_image, gt=train_gt, pred=train_pred)
                wandb.log({"training images": wandb.Image(fig)})
                # fig.savefig(f"train_fig/train_fig_{epoch_id}_{batch_id}")
                plt.close(fig)

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
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict, batch_id: int, epoch_id: int) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target_bis = copy.deepcopy(target)
                target_bis[target_bis == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target_bis = copy.deepcopy(target)[:, :-1]
        else:
            mask = None
            target_bis = copy.deepcopy(target)

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_bis, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
            
        if batch_id == 0: 
            val_image= data[0].detach().cpu().squeeze().float().numpy()
            # Deal with region-based training
            if target[0].shape[0] == 1:
                # Not region-based, just squeeze the class dimension
                val_gt = target.detach().cpu().squeeze().float().numpy()[0]
                val_pred = np.argmax(output.detach().cpu().squeeze().numpy(), axis=1)[0]                
            else:
                # Region-based, we need to sum all classes along the class dimension
                val_gt = target.detach().cpu().squeeze().float().numpy()[0]
                val_gt = np.squeeze(np.sum(val_gt, axis=0))  # Sum across classes to get a single mask and squeeze it
                val_pred = output.detach().cpu().squeeze().float().numpy()[0]
                val_pred = np.squeeze(np.sum(val_pred, axis=0))  # Sum across classes to get a single mask and squeeze it
            fig = plot_single_slice(combined=val_image, gt=val_gt, pred=val_pred)
            wandb.log({"Validation images": wandb.Image(fig)})
            # fig.savefig(f"val_fig/val_fig_{epoch_id}_{batch_id}")
            plt.close(fig)

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


def plot_single_slice(combined, gt, pred, debug=False):
    """
    Plot the image, ground truth, and prediction for a single slice.
    Assumes 2D inputs (H, W).
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Image | Ground Truth | Prediction')

    axs[0].imshow(combined.T, cmap='gray')
    axs[0].set_title("Image")
    axs[0].axis('off')

    axs[1].imshow(gt.T, cmap='Reds')
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(pred.T, cmap='Blues')
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    plt.tight_layout()
    return fig