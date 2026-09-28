"""Offline knowledge-distillation trainer (3D).

Student = a (smaller) 3D nnU-Net trained against the frozen teacher's cached soft logits PLUS
the real labels. Subclasses `nnUnet3DCustomTrainer` so the student inherits the exact winning
recipe (patch, optimizer, GPU spatial aug, default nnU-Net CPU augmentation, ...); only the
teacher term and the (separately-specified, smaller) architecture differ.

Additions vs. the parent:
  * dataloader: `nnUNetDistillDataLoader` also serves the cached teacher logit map, routed as a
    `regression_target` so it follows the CPU spatial/mirror transforms (linear interp) and is
    skipped by intensity transforms.
  * GPU augmentation: `GPU3DSpatialAugmentationKD` applies the same sampled flip+affine to data,
    target AND teacher (teacher with bilinear).
  * loss: L = SUP_WEIGHT * (Dice+CE vs GT) + KD_WEIGHT * T^2 * KL(student/T || teacher/T),
    at every deep-supervision scale (teacher logits trilinear-downsampled to each scale).

Config keys (via $NNUNET_EXP_CONFIG json, on top of the parent's keys), all overridable by an
env var of the same name too (handy for sweeps -- see scripts/02_train/03_run_all_conditions.sh):
  KD_TEMPERATURE (float, default 2.0), KD_WEIGHT (float, default 1.0),
  SUP_WEIGHT (float, default 1.0), TEACHER_LOGITS_DIR (str, dir name beside the preprocessed
  dataset, default 'teacher_logits__patchsize_5_adamw').

Installed flat into the venv `variants/` dir (see scripts/02_train/01_install_trainer.sh).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

from .nnUnet3DCustomTrainer import nnUnet3DCustomTrainer
from .distill_dataloader import nnUNetDistillDataLoader
from .distill_augmentation import GPU3DSpatialAugmentationKD
from .distill_loss import KDLoss, DeepSupervisionDistillation


class nnUnet3DDistillTrainer(nnUnet3DCustomTrainer):

    # ---- config ----
    def _apply_experiment_config(self):
        self.KD_TEMPERATURE = 2.0
        self.KD_WEIGHT = 1.0
        self.SUP_WEIGHT = 1.0
        self.TEACHER_LOGITS_DIR = "teacher_logits__patchsize_5_adamw"

        # the parent only knows its own keys, so it will (harmlessly) warn about ours below
        super()._apply_experiment_config()

        cfg_path = os.environ.get("NNUNET_EXP_CONFIG")
        cfg = json.load(open(cfg_path)) if cfg_path and os.path.isfile(cfg_path) else {}
        if "KD_TEMPERATURE" in cfg:
            self.KD_TEMPERATURE = float(cfg["KD_TEMPERATURE"])
        if "KD_WEIGHT" in cfg:
            self.KD_WEIGHT = float(cfg["KD_WEIGHT"])
        if "SUP_WEIGHT" in cfg:
            self.SUP_WEIGHT = float(cfg["SUP_WEIGHT"])
        if "TEACHER_LOGITS_DIR" in cfg:
            self.TEACHER_LOGITS_DIR = cfg["TEACHER_LOGITS_DIR"]
        if "num_iterations_per_epoch" in cfg:
            self.num_iterations_per_epoch = int(cfg["num_iterations_per_epoch"])
        if "num_val_iterations_per_epoch" in cfg:
            self.num_val_iterations_per_epoch = int(cfg["num_val_iterations_per_epoch"])

        # env vars win over the config file -- used to sweep KD_TEMPERATURE across conditions
        if os.environ.get("KD_TEMPERATURE"):
            self.KD_TEMPERATURE = float(os.environ["KD_TEMPERATURE"])
        if os.environ.get("KD_WEIGHT"):
            self.KD_WEIGHT = float(os.environ["KD_WEIGHT"])
        if os.environ.get("TEACHER_LOGITS_DIR"):
            self.TEACHER_LOGITS_DIR = os.environ["TEACHER_LOGITS_DIR"]

    def _teacher_folder(self) -> str:
        if os.path.isabs(self.TEACHER_LOGITS_DIR):
            return self.TEACHER_LOGITS_DIR
        return os.path.join(self.preprocessed_dataset_folder_base, self.TEACHER_LOGITS_DIR)

    # ---- dataloaders: train loader also serves teacher logits ----
    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        teacher_folder = self._teacher_folder()
        self.print_to_log_file(f"[distill] teacher logits dir: {teacher_folder}")
        self.print_to_log_file(f"[distill] KD_TEMPERATURE={self.KD_TEMPERATURE} "
                                f"KD_WEIGHT={self.KD_WEIGHT} SUP_WEIGHT={self.SUP_WEIGHT}")

        dl_tr = nnUNetDistillDataLoader(
            dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling, teacher_folder=teacher_folder)
        # validation only needs the supervised loss -> standard loader, no teacher logits
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size, self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == 'cuda', wait_time=0.002)
        next(mt_gen_train)
        next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    # ---- training lifecycle: swap in the KD-aware GPU augmentation + build the KD loss ----
    def on_train_start(self):
        super().on_train_start()

        base_aug = self.gpu_augmentation
        if base_aug is not None:
            kd_aug = GPU3DSpatialAugmentationKD(patch_size=base_aug.patch_size)
            for key, value in base_aug.__dict__.items():
                if not key.startswith('_'):
                    setattr(kd_aug, key, value)
            self.gpu_augmentation = kd_aug

        if self.enable_deep_supervision:
            ds_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))])
            weights[-1] = 0
            weight_factors = list(weights / weights.sum())
        else:
            weight_factors = None
        self.kd_loss = DeepSupervisionDistillation(KDLoss(self.KD_TEMPERATURE), weight_factors)
        self.kd_loss.to(self.device)

        self._sup_sum, self._kd_sum, self._kd_n = 0.0, 0.0, 0
        self.print_to_log_file(
            f"[distill] KD ready: T={self.KD_TEMPERATURE}, w_kd={self.KD_WEIGHT}, w_sup={self.SUP_WEIGHT}")

    def on_epoch_end(self):
        super().on_epoch_end()
        if self._kd_n > 0:
            mean_sup = self._sup_sum / self._kd_n
            mean_kd = self._kd_sum / self._kd_n
            self.print_to_log_file(
                f"[distill] mean train Sup_Loss={mean_sup:.4f} KD_Loss={mean_kd:.4f} "
                f"(weighted: {self.SUP_WEIGHT * mean_sup:.4f} + {self.KD_WEIGHT * mean_kd:.4f})")
            self._sup_sum, self._kd_sum, self._kd_n = 0.0, 0.0, 0

    def _downsample_teacher_on_gpu(self, teacher: torch.Tensor) -> list:
        ds_scales = self._get_deep_supervision_scales()
        downsampled = [teacher]
        for scale in ds_scales[1:]:
            new_shape = [int(teacher.shape[i + 2] * scale[i]) for i in range(3)]
            downsampled.append(F.interpolate(teacher, size=new_shape, mode='trilinear', align_corners=False))
        return downsampled

    # ---- training step: supervised + distillation ----
    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        teacher = batch['teacher'].to(self.device, non_blocking=True)
        target = batch['target']
        target = (target[0] if isinstance(target, list) else target).to(self.device, non_blocking=True)

        if self.EXP_SPATIAL_AUGMENTATION and self.gpu_augmentation is not None:
            data, target, teacher = self.gpu_augmentation(data, target, teacher)

        if self.enable_deep_supervision:
            target_list = self._downsample_target_on_gpu(target)
            teacher_list = self._downsample_teacher_on_gpu(teacher)
        else:
            target_list, teacher_list = target, teacher

        self.optimizer.zero_grad()
        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)
            l_sup = self.loss(output, target_list)
            if self.KD_WEIGHT != 0:
                l_kd = self.kd_loss(output, teacher_list, target_list)
                loss = self.SUP_WEIGHT * l_sup + self.KD_WEIGHT * l_kd
            else:  # control run: pure supervised, skip KD compute
                l_kd = torch.zeros((), device=l_sup.device)
                loss = self.SUP_WEIGHT * l_sup

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        sup_value, kd_value = float(l_sup.detach()), float(l_kd.detach())
        self._sup_sum += sup_value
        self._kd_sum += kd_value
        self._kd_n += 1
        if self.local_rank == 0:
            try:
                import wandb
                wandb.log({"Train/Sup_Loss": sup_value, "Train/KD_Loss": kd_value}, step=self.current_epoch)
            except Exception:
                pass

        return {'loss': loss.detach().cpu().numpy()}
