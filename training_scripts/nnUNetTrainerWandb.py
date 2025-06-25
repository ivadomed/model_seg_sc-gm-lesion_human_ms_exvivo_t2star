
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

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWandb(nnUNetTrainer):

    def run_training(self):
        self.on_train_start()
        output_path = os.path.join("output_path", str(datetime.now().date()) +"_" +str(datetime.now().time()))
        os.makedirs(output_path, exist_ok=True)

        wandb.init(project=f'model_seg_sc-gm-lesion_human_ms_exvivo_t2star',  dir=output_path)

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train),batch_id))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
        wandb.finish()  
        self.on_train_end()

    def train_step(self, batch: dict, batch_id: int) -> dict:
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
                train_gt= target[0].detach().cpu().squeeze().float().numpy()[0]
                train_pred = np.argmax(output[0].detach().cpu().squeeze().numpy(), axis=1)[0]                
                fig = plot_single_slice(combined=train_image, gt=train_gt, pred=train_pred)
                wandb.log({"training images": wandb.Image(fig)})
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