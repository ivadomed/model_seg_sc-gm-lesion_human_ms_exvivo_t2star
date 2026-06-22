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
from .augmentation_3D import GPU3DSpatialAugmentation
from .visualization_3D import decode_bitmask_to_multichannel, plot_3d_snapshot
from .preprocessing_3D import preprocess_data_gpu_3d

torch.set_float32_matmul_precision('high')

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

		self.EXP_SPATIAL_AUGMENTATION = True
		self.EXP_SPATIAL_AUGMENTATION_ID = 12
		self.EXP_SOFT_EDGE_LOSS_1 = False
		self.EXP_SOFT_EDGE_LOSS_2 = False
		self.EXP_SOFT_EDGE_LOSS_3 = False
		## For EXP_SGD_OPTIMIZER please comment/uncomment as needed in the configure_optimizers function (done like this for simplicity)
		self.EXP_SGD_OPTIMIZER = False

		active_exps = [self.EXP_MAG_ONLY_DUPLICATE, self.EXP_MAG_ONE_CHANNEL, self.EXP_OTSU, self.EXP_MAG_PREPRO, self.EXP_PHASE_PREPRO, 
						self.EXP_SPATIAL_AUGMENTATION,
						self.EXP_SOFT_EDGE_LOSS_1, self.EXP_SOFT_EDGE_LOSS_2, self.EXP_SOFT_EDGE_LOSS_3]
		
		if sum(active_exps) > 1: raise RuntimeError("Multiple experiment flags set to True.")

		self.configuration = configuration
		self.plans = plans
		self.wandb_project = "MagPhase_MRI_Seg_Noise_Transfer"
		self.wandb_run_name = f"Fold{fold}_{configuration}_{datetime.now().strftime('%Y%m%d_%H%M')}"
		
		if self.EXP_SPATIAL_AUGMENTATION: self.wandb_run_name += f"_spatial_aug_{self.EXP_SPATIAL_AUGMENTATION_ID}_3d"
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

		# --- Config-driven overrides (single source of truth) ---
		# The defaults above reproduce the historical behaviour. A config only overrides what it names,
		# via $NNUNET_EXP_CONFIG (path to a JSON) and/or $NNUNET_NUM_EPOCHS (int, handy for smoke tests).
		self._apply_experiment_config()

	def _apply_experiment_config(self):
		import json as _json
		cfg = {}
		cfg_path = os.environ.get("NNUNET_EXP_CONFIG")
		if cfg_path and os.path.isfile(cfg_path):
			with open(cfg_path) as _f:
				cfg = _json.load(_f)
			print(f"[nnUnet3DCustomTrainer] applying experiment config: {cfg_path}")
		# allowed keys -> attribute names (only these may be overridden)
		_allowed = {
			"num_epochs", "initial_lr",
			"EXP_MAG_ONLY_DUPLICATE", "EXP_MAG_ONE_CHANNEL", "EXP_OTSU", "EXP_MAG_PREPRO", "EXP_PHASE_PREPRO",
			"EXP_SPATIAL_AUGMENTATION", "EXP_SPATIAL_AUGMENTATION_ID",
			"EXP_SOFT_EDGE_LOSS_1", "EXP_SOFT_EDGE_LOSS_2", "EXP_SOFT_EDGE_LOSS_3", "EXP_SGD_OPTIMIZER",
		}
		for k, v in cfg.items():
			if k in _allowed:
				setattr(self, k, v)
			elif k not in ("name", "channels", "patch_size", "dataset_id", "family"):
				print(f"[nnUnet3DCustomTrainer] WARNING: ignoring unknown config key '{k}'")
		# quick epochs override (smoke tests)
		_ep = os.environ.get("NNUNET_NUM_EPOCHS")
		if _ep:
			self.num_epochs = int(_ep)
			print(f"[nnUnet3DCustomTrainer] num_epochs overridden via env -> {self.num_epochs}")
		# re-apply derived rule and re-validate single-active-experiment invariant
		if self.EXP_PHASE_PREPRO or self.EXP_MAG_PREPRO:
			self.EXP_OTSU = True
		_active = [self.EXP_MAG_ONLY_DUPLICATE, self.EXP_MAG_ONE_CHANNEL, self.EXP_OTSU, self.EXP_MAG_PREPRO,
				self.EXP_PHASE_PREPRO, self.EXP_SPATIAL_AUGMENTATION,
				self.EXP_SOFT_EDGE_LOSS_1, self.EXP_SOFT_EDGE_LOSS_2, self.EXP_SOFT_EDGE_LOSS_3]
		if sum(bool(x) for x in _active) > 1:
			raise RuntimeError("Multiple experiment flags active after config override.")

	def configure_optimizers(self):
		if self.EXP_SGD_OPTIMIZER:
			optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
										momentum=0.99, nesterov=True)
		else:
			optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
		lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.96)
		# lr_scheduler = CosineAnnealingLR(
			# optimizer, T_max=self.num_epochs, eta_min=0.00001, last_epoch=-1)
		return optimizer, lr_scheduler 
	
	def on_train_start(self):
		# 1. Initialize Standard Components (Network, Optimizer, etc.)
		super().on_train_start()
		
		self.optimizer, self.lr_scheduler = self.configure_optimizers()
		# self.network_complied = torch.compile(self.network)
  
		if self.local_rank == 0:
			config = {
				"fold": self.fold,
				"configuration": self.configuration_name,
				"batch_size": self.configuration_manager.batch_size,
				f"EXP_SPATIAL_AUGMENTATION_{self.EXP_SPATIAL_AUGMENTATION_ID}": self.EXP_SPATIAL_AUGMENTATION,
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
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 1:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(30, 30, 30),
				scale_range=(0.7, 1.4),
				p_affine=0.3,
				p_flip=0.5
			).to(self.device)
		
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 2:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(180, 180, 180),
				scale_range=(0.6, 1.5),
				p_affine=0.4,
				p_flip=0.5
			).to(self.device)

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 3:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(90, 90, 90),
				scale_range=(0.8, 1.2),
				p_affine=0.3,
				p_flip=0.5
			).to(self.device)
			
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 4:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 180, 0),
				scale_range=(0.8, 1.2),
				p_affine=0.3,
				p_flip=0.5
			).to(self.device)
		
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 5:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 180, 0),
				scale_range=(0.4, 1.8),
				p_affine=0.7,
				p_flip=0.5
			).to(self.device)
			
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 6:
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

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 7:
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

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 8:
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

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 9:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(90, 90, 90),
				scale_range=(0.7, 1.7),
				trans_range=(0.45, 0.45, 0.45),
				shear_range_deg=(35, 35, 35, 35, 35, 35), 
				persp_factor=0.35, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 10:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(180, 180, 180),
				scale_range=(0.1, 3.0),
				trans_range=(0.8, 0.8, 0.8),
				shear_range_deg=(85, 85, 85, 85, 85, 85),
				persp_factor=0.85, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 11:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(180, 180, 180),
				scale_range=(0.3, 2.0),
				trans_range=(0.45, 0.45, 0.45),
				shear_range_deg=(55, 55, 55, 55, 55, 55), 
				persp_factor=0.55, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)   

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 12:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(180, 180, 180),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device) 
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 13:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 180, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 14:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(180, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 15:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 180),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 16:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.45, 0.45, 0.45),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 17:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.45, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 18:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.45, 0.0, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 19:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.45),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 20:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(180, 0, 180),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 21:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 180, 180),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 22:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(180, 180, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 23:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.45, 0.0, 0.45),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 24:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.45, 0.45, 0.0),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)

		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 25:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.45, 0.45),
				shear_range_deg=(0, 0, 0, 0, 0, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
				p_affine=0.95,
				p_flip=0.5
			).to(self.device)
   
		if self.EXP_SPATIAL_AUGMENTATION and self.EXP_SPATIAL_AUGMENTATION_ID == 26:
			self.gpu_augmentation = GPU3DSpatialAugmentation(
				patch_size=self.configuration_manager.patch_size,
				rot_range_deg=(0, 0, 0),
				scale_range=(1.0, 1.0),
				trans_range=(0.0, 0.0, 0.0),
				shear_range_deg=(0, 35, 0, 0, 35, 0), 
				persp_factor=0.0, 
				keep_y_parallel=False,
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
			
		if self.EXP_SPATIAL_AUGMENTATION and self.gpu_augmentation is not None:
			data, target = self.gpu_augmentation(data, target)

		if self.EXP_OTSU:
			data = preprocess_data_gpu_3d(data, mag_prepro=self.EXP_MAG_PREPRO, phase_prepro=self.EXP_PHASE_PREPRO)

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
			data = preprocess_data_gpu_3d(data, mag_prepro=self.EXP_MAG_PREPRO, phase_prepro=self.EXP_PHASE_PREPRO)
		
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