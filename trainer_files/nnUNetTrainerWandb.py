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
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

from skimage.restoration import unwrap_phase
from skimage.exposure import equalize_adapthist as clahe, rescale_intensity

from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from nnunetv2.training.loss.dice import SoftDiceLoss, get_tp_fp_fn_tn


from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from typing import Union, List, Tuple, Dict
import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.noise.rician import RicianNoiseTransform
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, binary_opening
import kornia 


from .custom_loss import DC_and_CE_with_Edge_Loss
from .custom_loss import DeepSupervisionWrapper
from .preprocessing_2D import _preprocess_data_gpu, GPUPreprocessingTransform
from .augmentation_2D import CustomSpatialTransform

DO_PREPROCESSING = True

class nnUNetTrainerWandb(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_segmentation_classes = len(self.label_manager.foreground_regions) if self.label_manager.has_regions else len(self.label_manager.foreground_labels)
        self.num_segmentation_classes += 1 ## because of the background class        

        # Experiment flags
        ### IMPORTANT: Most likely due to torch compiling + nnunet backend process, the exp_base_no_otsu flag doesn't work, i believe the if/else statement is broken by the compiling
        ### Instead the preprocessing should be commented out.
        self.exp_base_no_otsu = False
        
        self.exp_mag_one_channel = False
        self.exp_base = False
        self.exp_soft_loss = True
        self.exp_mosaic = False
        
        # Determine mag_only behavior: 
        # If true single channel (from prepro), we don't need this flag to force copy.
        # This flag was likely for 2-channel data where you wanted to simulate 1 channel.
        self.exp_mag_only = False 
        
        self.exp_phase_prepro = False
        self.exp_mag_prepro = False
        ### IMPORTANT: Most likely due to torch compiling + nnunet backend process, the exp_no_spatial_aug flag doesn't work, i believe the if/else statement is broken by the compiling
        ### Instead the spatial transformation should be commented out.
        self.exp_no_spatial_aug = False
        self.exp_spatial_aug_2 = True
        self.exp_spatial_aug_3 = False
        self.exp_SGD = False

        # Ensure exactly one experiment is active
        # (Modified logic to exclude exp_mag_only if we are natively single channel, to avoid confusion)
        active_experiments = [name for name, value in self.__dict__.items() if name.startswith('exp_') and value]
        if len(active_experiments) != 2:
            raise ValueError(f"There must be exactly one active experiment, but found {len(active_experiments)}: {active_experiments}")
        
        self.active_experiment = active_experiments[0]
        if len(active_experiments) == 2:
            self.active_experiment_2 = active_experiments[1]
        else: 
            self.active_experiment_2 = ""
        
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()


        self.data_aug = self.get_training_transforms(

            patch_size=patch_size, rotation_for_DA=rotation_for_DA, deep_supervision_scales=deep_supervision_scales, mirror_axes=mirror_axes, do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
            do_preprocessing=DO_PREPROCESSING,
            exp_no_spatial_aug=self.exp_no_spatial_aug,
            exp_mag_prepro=self.exp_mag_prepro,
            exp_phase_prepro=self.exp_phase_prepro,
            no_otsu=self.exp_base_no_otsu
            )
        
    
    def _get_augmentation_config(self) -> dict:
        """
        Parses the augmentation pipeline to create a loggable dictionary.
        This updated version handles nested transforms like `RandomTransform`.
        """
        # Check if data_aug exists and is a Compose or LimitedLenWrapper instance
        if not hasattr(self, 'data_aug'):
            return {}

        if isinstance(self.data_aug, ComposeTransforms):
            transforms_to_log = self.data_aug.transforms
        else:
            return {}

        aug_config = {}
        for t in transforms_to_log:
            params = {}
            # Handle RandomTransform by extracting the nested transform and its probability
            if isinstance(t, RandomTransform):
                # The actual transform is nested inside
                actual_transform = t.transform
                transform_name = type(actual_transform).__name__
                # Add apply_probability from the wrapper
                params['apply_probability'] = t.apply_probability
            else:
                actual_transform = t
                transform_name = type(actual_transform).__name__

            # --- Parameter extraction for each transform type ---

            if isinstance(actual_transform, SpatialTransform):
                params.update({
                    'patch_size': actual_transform.patch_size,
                    'p_elastic_deform': actual_transform.p_elastic_deform,
                    'p_rotation': actual_transform.p_rotation,
                    'p_scaling': actual_transform.p_scaling,
                    'random_crop': actual_transform.random_crop,
                })
            elif isinstance(actual_transform, MirrorTransform):
                params['allowed_axes'] = actual_transform.allowed_axes
            elif isinstance(actual_transform, GaussianNoiseTransform):
                params['noise_variance'] = actual_transform.noise_variance
            elif isinstance(actual_transform, GaussianBlurTransform):
                params['blur_sigma'] = actual_transform.blur_sigma
            elif isinstance(actual_transform, MultiplicativeBrightnessTransform):
                # Logging its string representation is a safe way to capture its config
                params['multiplier_range'] = str(actual_transform.multiplier_range)
            elif isinstance(actual_transform, ContrastTransform) :
                params['contrast_range'] = str(actual_transform.contrast_range)
            elif isinstance(actual_transform, SimulateLowResolutionTransform):
                params['scale'] = actual_transform.scale
            elif isinstance(actual_transform, GammaTransform):
                # Similar to contrast, gamma range can be complex
                params['gamma_range'] = str(getattr(actual_transform, 'gamma', 'N/A'))
            elif isinstance(actual_transform, RicianNoiseTransform):
                params['noise_variance'] = actual_transform.noise_variance
            # Add other specific transforms without parameters if needed for logging presence
            elif isinstance(actual_transform, (Convert3DTo2DTransform, Convert2DTo3DTransform,
                                                MaskImageTransform, RemoveLabelTansform,
                                                ConvertSegmentationToRegionsTransform,
                                                DownsampleSegForDSTransform)):
                pass  # No complex parameters to log, just the name is enough

            # Use a unique key for each transform in case of multiple transforms of the same type
            count = 1
            log_name = transform_name
            while log_name in aug_config:
                count += 1
                log_name = f"{transform_name}_{count}"
            
            aug_config[log_name] = params
        
        return aug_config
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    # momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.96)
        # lr_scheduler = CosineAnnealingLR(
            # optimizer, T_max=self.num_epochs, eta_min=0.00001, last_epoch=-1)
        return optimizer, lr_scheduler 


    def on_train_start(self):
        """
        Initializes the weighted loss function. This is the correct place to do it,
        ensuring the weight tensor is created on the final target device.
        """
        self.device = torch.device('cuda')
        self.enable_deep_supervision = True
        super().on_train_start() 
        
        self.dataloader_train.num_processes = 16
        self.dataloader_val.num_processes = 16
    
        self.save_every = 20
        self.num_epochs = 200
        self.initial_lr = 1e-3
        
        # Default values
        self.mosaic_probability = 0.0
        self.edge_params = {}

        if self.exp_soft_loss:
            #EXP_SOFT_LOSS_1
            # self.edge_params = {
            #     1: {'edge_weight': 0.9, 'kernel_size': 7},
            #     2: {'edge_weight': 0.9, 'kernel_size': 3},  
            #     3: {'edge_weight': 0.6, 'kernel_size': 5},
            #     4: {'edge_weight': 0.4, 'kernel_size': 7}
            # }
            #EXP_SOFT_LOSS_2
            self.edge_params = {
                1: {'edge_weight': 0.9, 'kernel_size': 7},
                2: {'edge_weight': 0.9, 'kernel_size': 3},  
                3: {'edge_weight': 0.6, 'kernel_size': 3},
                4: {'edge_weight': 0.4, 'kernel_size': 3}
            }
            #EXP_SOFT_LOSS_3
            # self.edge_params = {
            #     1: {'edge_weight': 0.7, 'kernel_size': 5},
            #     2: {'edge_weight': 0.6, 'kernel_size': 3},  
            #     3: {'edge_weight': 0.2, 'kernel_size': 3},
            #     4: {'edge_weight': 0.2, 'kernel_size': 3}
            # }
        
        if self.exp_mosaic:
            self.mosaic_probability = 1.0 

        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': False,
            'ddp': self.is_ddp
        }

        self.loss = DC_and_CE_with_Edge_Loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs={},
            weight_ce=6.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss, # As used in original setup
            edge_params=self.edge_params,
            blur_sigma=1.0
        )

        # Deep supervision wrapping remains the same
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            self.loss = DeepSupervisionWrapper(self.loss, weights)
        
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.wandb_init()
        
    
    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
            do_preprocessing: bool = True,
            exp_no_spatial_aug=False,
            exp_mag_prepro=False,
            exp_phase_prepro=False,
            no_otsu=False
    ) -> BasicTransform:
        print("IN THE CUSTOM GET TRAINING")
        transforms = []
        

        if do_preprocessing : 
            transforms.append(GPUPreprocessingTransform(mag_prepro=exp_mag_prepro, phase_prepro=exp_phase_prepro, no_otsu=no_otsu))
        
        # transforms.append(CPUPreprocessingTransform())
            
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        
        
        transforms.append(
            SpatialTransform(
                patch_size=patch_size_spatial,  
                patch_center_dist_from_border=0,
                random_crop=False,             
                p_elastic_deform=0,
                elastic_deform_scale=(0.2, 0.8),           
                p_rotation=0,              
                p_scaling=0,
            )
        )
        
        
        #### IMPORTANT: to run the no spatial aug exp you'll need to comment this whole if statement out (because the compiling or something else in the backend of nnunet) breaks it.
        ### EXP_SPTAIAL_AUGMENTATION_1
        # max_translate_dist = [int(p * 0.45) for p in patch_size_spatial]
        # transforms.append(
        #     CustomSpatialTransform(
        #         patch_size=patch_size_spatial,
        #         p_per_sample=0.95,     
        #         degrees=(-90, 90),     
        #         scale=(0.7, 1.7),     
        #         translation_px=tuple(max_translate_dist),
        #         shear=(-35, 35),       
        #         perspective=0.35       
        #     )
        # )
        #EXP_SPTAIAL_AUGMENTATION_2
        max_translate_dist = [int(p * 0.8) for p in patch_size_spatial]
        transforms.append(
            CustomSpatialTransform(
                patch_size=patch_size_spatial,
                p_per_sample=0.95,     
                degrees=(-180, 180),     
                scale=(0.3, 2.0),     
                translation_px=tuple(max_translate_dist),
                shear=(-55, 55),       
                perspective=0.55       
            )
        )
        #EXP_SPTAIAL_AUGMENTATION_3
        # max_translate_dist = [int(p * 0.8) for p in patch_size_spatial]
        # transforms.append(
        #     CustomSpatialTransform(
        #         patch_size=patch_size_spatial,
        #         p_per_sample=0.95,     
        #         degrees=(-180, 180),     
        #         scale=(0.1, 3.0),     
        #         translation_px=tuple(max_translate_dist),
        #         shear=(-85, 85),       
        #         perspective=0.85       
        #     )
        # )
        
        transforms.append(MirrorTransform((0,1)))

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=False
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
                
        return ComposeTransforms(transforms)
    
    def run_training(self):
        self.on_train_start()
        
        for epoch in tqdm.tqdm(range(self.current_epoch, self.num_epochs)):
            # if epoch == 25 : 
            #     raise ValueError("Stopping training at epoch 25 for testing purposes.")
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in tqdm.tqdm(range(self.num_iterations_per_epoch)):
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
        """Initializes a new wandb run with a comprehensive configuration."""
        project_name = "nnUNet_v2_human_ms_exvivo_t2star"
        
        # --- Build a comprehensive config dictionary ---
        config = {
            # Experiment
            "experiment": self.active_experiment,
            "experiment 2": self.active_experiment_2,
            # High-level info
            "plans_name": self.plans_manager.plans_name, "configuration": self.configuration_name, "fold": self.fold,
            "dataset_name": self.plans_manager.dataset_name, "timestamp": datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            
            # Core training parameters
            "num_epochs": self.num_epochs, "batch_size": self.batch_size, "patch_size": self.configuration_manager.patch_size,
            "num_segmentation_classes": self.num_segmentation_classes, "initial_lr": self.initial_lr,
            "deep_supervision": self.enable_deep_supervision,
            
            # Optimizer
            "optimizer": type(self.optimizer).__name__,
            "optimizer_params": {k: v for k, v in self.optimizer.defaults.items() if k != 'lr'}, # lr is logged separately per epoch
            
            # Scheduler
            "scheduler": type(self.lr_scheduler).__name__,
            
            # Loss Function
            "loss_function": type(self.loss).__name__,
            
            # Preprocessing - Extracted from _preprocess_data method
            "preprocessing": {
                'magnitude': {'method': 'CLAHE', 'normalization': 'min-max to [0, 1]', 'clip_limit': 0.01},
                'phase': {'method': 'Unwrap and Rescale Intensity', 'unwrap_function': 'skimage.restoration.unwrap_phase', 'rescale_percentiles': (30, 75)}
            },
            
            # Augmentations - Automatically parsed
            "augmentations": self._get_augmentation_config(),
            
            # The full plans file is large but very useful for reproducibility
            "plans": self.plans_manager.plans
        }
        if not DO_PREPROCESSING:
            config['preprocessing'] = {
                'magnitude': {'method': 'None'},
                'phase': {'method': 'None'}
            }
        

        # Add loss-specific parameters
        # if isinstance(self.loss, DC_and_CE_loss):
        try : 
            config['loss_params'] = self.loss.apply_kwargs
        except :
            pass
        # elif 'WeightedSoftDiceLoss' in str(type(self.loss)):
        try : 
            # if hasattr(self.loss, 'weight'):
            config['loss_params'] = {'channel_weights': self.loss.weight.cpu().numpy().tolist()}
        except : 
            pass
        
        # elif isinstance(self.loss, DC_and_CE_with_Edge_Loss):
        try :
            config['loss_params']  = {'edge_params': self.loss.edge_params, 'weight_ce': self.loss.weight_ce, 'weight_dice': self.loss.weight_dice}
        except : 
            pass

        wandb.init(
            project=project_name,
            name=f"{self.configuration_name}_fold{self.fold}_{config['timestamp']}",
            config=config
        )
        wandb.watch(self.network, log="all", log_freq=self.num_iterations_per_epoch)

        
    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        avg_total_loss = np.mean([i['loss'] for i in train_outputs])
        avg_dice_loss = np.mean([i['dice_loss'] for i in train_outputs])
        avg_ce_loss = np.mean([i['ce_loss'] for i in train_outputs])

        wandb.log({
            "Training Curves/Train_Loss": avg_total_loss,
            "Training Curves/Train_Dice_Loss": avg_dice_loss,
            "Training Curves/Train_CE_Loss": avg_ce_loss
        }, step=self.current_epoch)
        
                
        # avg_train_loss = np.mean([i['loss'] for i in train_outputs])
        # wandb.log({"Training Curves/Train_Loss": avg_train_loss}, step=self.current_epoch)

        
    def on_epoch_end(self):
        super().on_epoch_end()
        wandb.log({"Hyperparameters/Learning_Rate": self.optimizer.param_groups[0]['lr']}, step=self.current_epoch)

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)

        mean_fg_dice = self.logger.my_fantastic_logging['mean_fg_dice'][-1]
        # val_loss = self.logger.my_fantastic_logging['val_losses'][-1]
        
        # wandb_log_dict = {
        #     "Training Curves/Val_Mean_Foreground_Dice": mean_fg_dice,
        #     "Training Curves/Val_Loss": val_loss,
        # }



        val_total_loss = self.logger.my_fantastic_logging['val_losses'][-1] 

        avg_val_dice_loss = np.mean([i['val_dice_loss'] for i in val_outputs])
        avg_val_ce_loss = np.mean([i['val_ce_loss'] for i in val_outputs])

        wandb_log_dict = {
            "Training Curves/Val_Mean_Foreground_Dice": mean_fg_dice,
            "Training Curves/Val_Loss": val_total_loss,
            "Training Curves/Val_Dice_Loss": avg_val_dice_loss, 
            "Training Curves/Val_CE_Loss": avg_val_ce_loss,    
        }
        
        dice_per_class_or_region = self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]
        if self.label_manager.has_regions:
            for i, name in enumerate(self.label_manager.region_names):
                wandb_log_dict[f'Validation_Dice/{name}'] = dice_per_class_or_region[i]
        else:
            for i, label_id in enumerate(self.label_manager.foreground_labels+[5]):
                # ####### FOR THE LESION ONLY TRAINING, TO LOG CORRECTLY ########
                # if label_id-1 == 1 : 
                #     label_id = 3+1
                # ####### FOR THE LESION ONLY TRAINING, TO LOG CORRECTLY (END) ########
                wandb_log_dict[f'Validation_Dice/Class_{label_id-1}'] = dice_per_class_or_region[i]
        
        wandb.log(wandb_log_dict, step=self.current_epoch)


    def create_mosaic_batch(self, data: torch.Tensor, target: List[torch.Tensor], patch_size: Tuple[int, int] = (384, 384)) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Creates a 2x2 mosaic for high-resolution training.
        1. Resizes all input images/masks to a fixed patch_size (e.g., 384x384).
        2. Stitches four of these images into a large (e.g., 768x768) canvas.
        3. The final large mosaic is returned without being downsized.
        """
        # --- Configuration ---
        GRID_SIZE = 2
        IMGS_PER_MOSAIC = GRID_SIZE * GRID_SIZE # 4

        # --- Get shapes and device info ---
        batch_size, num_channels, _, _ = data.shape
        device = data.device
        target_full_res = target[0]

        # --- 1. Pad the batch if it's not a multiple of 4 ---
        num_mosaics = (batch_size + IMGS_PER_MOSAIC - 1) // IMGS_PER_MOSAIC
        needed_imgs = num_mosaics * IMGS_PER_MOSAIC
        padding_needed = needed_imgs - batch_size

        if padding_needed > 0:
            orig_h, orig_w = data.shape[2:]
            data_padding = torch.zeros((padding_needed, num_channels, orig_h, orig_w), dtype=data.dtype, device=device)
            data = torch.cat([data, data_padding], dim=0)
            
            target_padding = torch.zeros((padding_needed, 1, orig_h, orig_w), dtype=target_full_res.dtype, device=device)
            target_full_res = torch.cat([target_full_res, target_padding], dim=0)
        
        # --- 2. Resize all source images to the target patch size ---
        data_resized = F.interpolate(data, size=patch_size, mode='bilinear', align_corners=False)
        target_resized = F.interpolate(target_full_res.float(), size=patch_size, mode='nearest').long()

        # --- 3. Assemble resized images into large mosaic grids ---
        large_mosaics_data = []
        large_mosaics_target = []

        for i in range(num_mosaics):
            start_idx = i * IMGS_PER_MOSAIC
            end_idx = start_idx + IMGS_PER_MOSAIC
            
            data_group = data_resized[start_idx:end_idx]
            target_group = target_resized[start_idx:end_idx]

            # Create the 2x2 grid (final size will be 2*h x 2*w)
            top_row_data = torch.cat([data_group[0], data_group[1]], dim=2)
            bottom_row_data = torch.cat([data_group[2], data_group[3]], dim=2)
            large_mosaic_data = torch.cat([top_row_data, bottom_row_data], dim=1)
            large_mosaics_data.append(large_mosaic_data)

            top_row_target = torch.cat([target_group[0], target_group[1]], dim=2)
            bottom_row_target = torch.cat([target_group[2], target_group[3]], dim=2)
            large_mosaic_target = torch.cat([top_row_target, bottom_row_target], dim=1)
            large_mosaics_target.append(large_mosaic_target)

        # --- 4. Assign the large mosaics directly as the final output ---
        # The previous downscaling step has been removed.
        final_data = torch.stack(large_mosaics_data)
        final_target_full_res = torch.stack(large_mosaics_target)

        # --- 5. Recreate Deep Supervision Targets from the large mosaic ---
        new_target_list = [final_target_full_res]
        if self.enable_deep_supervision and len(target) > 1:
            # The new dimensions for deep supervision are the full mosaic size
            mosaic_h, mosaic_w = final_data.shape[2:] 

            deep_supervision_scales = self._get_deep_supervision_scales()
            for scale in deep_supervision_scales[1:]:
                scaled_h = int(mosaic_h * scale[0])
                scaled_w = int(mosaic_w * scale[1])
                downsampled_target = F.interpolate(final_target_full_res.float(), size=(scaled_h, scaled_w), mode='nearest').long()
                new_target_list.append(downsampled_target)

        return final_data, new_target_list
    
    

    def train_step(self, batch: dict, batch_id: int, epoch_id: int) -> dict:
        data, target = batch['data'], batch['target']
        
        data = data.to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in target]
        
        # Only apply this hack if we have 2 channels but want to force mag-only behavior
        if self.exp_mag_only and not self.exp_mag_one_channel:
            data[:, 1, :, :] = data[:, 0, :, :]
        
        if random.random() < self.mosaic_probability:
            data, target = self.create_mosaic_batch(data, target)
            
        self.optimizer.zero_grad(set_to_none=True)
        # with autocast(enabled=True) if self.device.type == 'cuda' else dummy_context():        
        with autocast(enabled=True) : 
            output = self.network(data) 
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            self.optimizer.step()

        if batch_id == 0 and (self.current_epoch % 3 == 0 or self.current_epoch == self.num_epochs - 1):
            with torch.no_grad():
                output_for_plotting = output[0] if self.enable_deep_supervision else output
                target_for_plotting = target[0] if self.enable_deep_supervision else target
                
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
                
                    
                orig_mag = None 
                orig_phase = None 
                
                aug_mag = data[0, 0].cpu().numpy()
                
                # Safe access for phase
                aug_phase = data[0, 1].cpu().numpy() if not self.exp_mag_one_channel else None
                
                proc_mag = None 
                proc_phase = None 
            
            
                fig = plot_comparison_and_segmentation(
                    orig_mag=orig_mag, orig_phase=orig_phase,
                    aug_mag=aug_mag, aug_phase=aug_phase,
                    proc_mag=proc_mag, proc_phase=proc_phase,
                    gt_masks=train_gt_masks, 
                    pred_masks_full=train_pred_masks_full, 
                    pred_masks_simplified=train_pred_masks_simplified
                )
                
                if hasattr(self.loss, 'last_weight_map') and self.loss.last_weight_map is not None:
                    # Retrieve the map (it's from the primary resolution via DeepSupervisionWrapper)
                    edge_map_tensor = self.loss.last_weight_map
                    
                    # Get the first item in the batch and convert to a plottable numpy array
                    edge_map_np = edge_map_tensor[0].squeeze().cpu().numpy()
                    
                    # Use the preprocessed magnitude as the background image
                    mag = data[0, 0].cpu().numpy()
                    
                    edge_fig = plot_edge_map(
                        base_image=mag, 
                        edge_map=edge_map_np, 
                        epoch=self.current_epoch
                    )

                    # Log the figure to wandb
                    wandb.log({"Plots/Edge_Weight_Map_Train": wandb.Image(edge_fig)}, step=self.current_epoch)
                    plt.close(edge_fig)
                    
                wandb.log({"Plots/Segmentation_Comparison_Train": wandb.Image(fig)}, step=self.current_epoch)
                plt.close(fig)
        
        loss_dict = {
            'loss': l.detach().cpu().numpy(),
            'dice_loss': self.loss.dice_loss.cpu().detach().numpy(),
            'ce_loss': self.loss.ce_loss.cpu().detach().numpy()
        }
        return loss_dict
    
    def validation_step(self, batch: dict, batch_id: int, epoch_id: int) -> dict:
        data, target = batch['data'], batch['target']
        
        data = data.to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in target]
        
        # Guard old hack
        if self.exp_mag_only and not self.exp_mag_one_channel:
            data[:, 1, :, :] = data[:, 0, :, :]
        
        if random.random() < self.mosaic_probability:
            data, target = self.create_mosaic_batch(data, target)
            
        original_data = data.clone()
        
        if self.exp_base_no_otsu : 
            preprocessed_data = data
        else:
            preprocessed_data = _preprocess_data_gpu(data, mag_prepro=self.exp_mag_prepro, phase_prepro=self.exp_phase_prepro)

        preprocessed_data = preprocessed_data.to(self.device, non_blocking=True)

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
                # Safe access for phase
                orig_phase = original_data[0, 1].cpu().numpy() if not self.exp_mag_one_channel else None
                
                proc_mag = preprocessed_data[0, 0].cpu().numpy()
                # Safe access for phase
                proc_phase = preprocessed_data[0, 1].cpu().numpy() if not self.exp_mag_one_channel else None
                
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

                if self.current_epoch % 3 == 0 or self.current_epoch == self.num_epochs - 1 :
                    fig = plot_comparison_and_segmentation(
                        orig_mag, orig_phase, proc_mag, proc_phase,
                        val_gt_masks, val_pred_masks_full, val_pred_masks_simplified
                    )
                    wandb.log({"Plots/Segmentation_Comparison_Val": wandb.Image(fig)}, step=self.current_epoch)
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
         
        return_dict = {
            'loss': l.detach().cpu().numpy(), # Base logger uses the 'loss' key
            'val_dice_loss': self.loss.dice_loss.cpu().numpy(),
            'val_ce_loss': self.loss.ce_loss.cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }
        return return_dict