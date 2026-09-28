"""Dataloader for offline distillation.

Subclasses nnU-Net's `nnUNetDataLoader` to additionally load the cached teacher logit map for
each case, crop it with the SAME bbox as image/seg, and route it through the augmentation
pipeline as `regression_target`. batchgeneratorsv2 transforms `regression_target` with the
image's spatial/mirror transforms (linear interp) and skips it for intensity transforms -- which
is exactly what a soft label needs. The transformed teacher map comes back under batch key
`teacher`; deep-supervision downsampling of it happens on-GPU in the trainer.

Installed flat into the venv `variants/` dir (see scripts/02_train/01_install_trainer.sh).
"""
import os

import numpy as np
import torch
import blosc2
from threadpoolctl import threadpool_limits
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class nnUNetDistillDataLoader(nnUNetDataLoader):
    def __init__(self, *args, teacher_folder: str = None, **kwargs):
        assert teacher_folder is not None, "nnUNetDistillDataLoader requires teacher_folder"
        assert os.path.isdir(teacher_folder), f"teacher_folder not found: {teacher_folder}"
        self.teacher_folder = teacher_folder
        super().__init__(*args, **kwargs)
        blosc2.set_nthreads(1)
        first_case = self._data.identifiers[0]
        probe = blosc2.open(urlpath=os.path.join(teacher_folder, first_case + ".b2nd"), mode="r")
        self.num_teacher_channels = probe.shape[0]

    def _load_teacher(self, identifier: str) -> np.ndarray:
        path = os.path.join(self.teacher_folder, identifier + ".b2nd")
        return np.asarray(blosc2.open(urlpath=path, mode="r")[:], dtype=np.float32)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        teacher_all = np.zeros((self.batch_size, self.num_teacher_channels, *self.patch_size), dtype=np.float32)

        for j, case_id in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, seg_prev, properties = self._data.load_case(case_id)
            teacher = self._load_teacher(case_id)
            assert teacher.shape[1:] == data.shape[1:], \
                f"{case_id}: teacher {teacher.shape} vs data {data.shape} spatial mismatch"

            bbox_lbs, bbox_ubs = self.get_bbox(data.shape[1:], force_fg, properties['class_locations'])
            bbox = [[a, b] for a, b in zip(bbox_lbs, bbox_ubs)]

            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped
            teacher_all[j] = crop_and_pad_nd(teacher, bbox, 0)  # pad logits with 0 (uniform)

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
            teacher_all = teacher_all[:, :, 0]

        if self.transforms is None:
            return {'data': data_all, 'target': seg_all, 'teacher': teacher_all, 'keys': selected_keys}

        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            data_all = torch.from_numpy(data_all).float()
            seg_all = torch.from_numpy(seg_all).to(torch.int16)
            teacher_all = torch.from_numpy(teacher_all).float()

            images, segs, teachers = [], [], []
            for b in range(self.batch_size):
                out = self.transforms(image=data_all[b], segmentation=seg_all[b], regression_target=teacher_all[b])
                images.append(out['image'])
                segs.append(out['segmentation'])
                teachers.append(out['regression_target'])

            data_all = torch.stack(images)
            teacher_all = torch.stack(teachers)
            if isinstance(segs[0], list):  # deep supervision: one seg map per scale
                seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
            else:
                seg_all = torch.stack(segs)

        return {'data': data_all, 'target': seg_all, 'teacher': teacher_all, 'keys': selected_keys}
