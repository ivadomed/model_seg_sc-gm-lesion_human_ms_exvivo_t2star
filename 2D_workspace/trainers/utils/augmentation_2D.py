from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from typing import Union, List, Tuple, Dict
import random
import kornia


class CustomSpatialTransform(BasicTransform):
    """
    A single, unified transform for all spatial augmentations.
    This version uses the correct Kornia API by creating separate augmenter
    instances for the image and mask to ensure proper interpolation.
    """
    def __init__(self,
                 patch_size: Tuple[int, ...],
                 p_per_sample: float = 1.0,
                 degrees: Tuple[float, float] = (-15, 15),
                 scale: Tuple[float, float] = (0.6, 1.4),
                 translation_px: Tuple[int, ...] = (20, 20),
                 shear: Tuple[float, float] = (-10, 10),
                 perspective: float = 0.1,
                 data_key: str = "image",
                 label_key: str = "segmentation"):
        super(CustomSpatialTransform, self).__init__()
        self.patch_size = patch_size
        self.p_per_sample = p_per_sample
        
        self.affine_augmenter_image = kornia.augmentation.RandomAffine(
            degrees=degrees, translate=tuple(float(t) / p for t, p in zip(translation_px, patch_size)),
            scale=scale, shear=shear, p=1.0, resample='bilinear'
        )
        self.perspective_augmenter_image = kornia.augmentation.RandomPerspective(
            distortion_scale=perspective, p=1.0, resample='bilinear'
        )

        # One for the mask with 'nearest' interpolation
        self.affine_augmenter_mask = kornia.augmentation.RandomAffine(
            degrees=degrees, translate=tuple(float(t) / p for t, p in zip(translation_px, patch_size)),
            scale=scale, shear=shear, p=1.0, resample='nearest'
        )
        self.perspective_augmenter_mask = kornia.augmentation.RandomPerspective(
            distortion_scale=perspective, p=1.0, resample='nearest'
        )

        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        if random.random() > self.p_per_sample:
            return data_dict

        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        # --- 1. Random Crop to Patch Size ---
        data_shape = data.shape[1:]
        if any(i < j for i, j in zip(data_shape, self.patch_size)):
             raise ValueError(f"Data shape {data_shape} is smaller than patch size {self.patch_size}")
        starts = [random.randint(0, i - j) for i, j in zip(data_shape, self.patch_size)]
        slicing = [slice(None)] + [slice(s, s + p) for s, p in zip(starts, self.patch_size)]
        cropped_data = data[tuple(slicing)].clone()
        if seg is not None:
            cropped_seg = seg[tuple(slicing)].clone()
        
        # --- 2. Generate Parameters ONCE ---
        # Generate random parameters using the image augmenter
        params_affine = self.affine_augmenter_image.generate_parameters(cropped_data.unsqueeze(0).shape)
        
        # --- 3. Apply Transformations to Image ---
        # Apply affine transform to image
        data_after_affine = self.affine_augmenter_image(cropped_data.unsqueeze(0), params=params_affine)
        
        # Generate perspective params and apply transform to image
        params_perspective = self.perspective_augmenter_image.generate_parameters(data_after_affine.shape)
        final_data = self.perspective_augmenter_image(data_after_affine, params=params_perspective).squeeze(0)
        data_dict[self.data_key] = final_data

        # --- 4. Apply IDENTICAL Transformations to Segmentation Mask ---
        if seg is not None:
            seg_for_transform = cropped_seg.float().unsqueeze(0)

            # Apply the SAME affine params using the MASK augmenter
            seg_after_affine = self.affine_augmenter_mask(
                seg_for_transform, params=params_affine
            )

            # Apply the SAME perspective params using the MASK augmenter
            final_seg_transformed = self.perspective_augmenter_mask(
                seg_after_affine, params=params_perspective
            )
            
            data_dict[self.label_key] = final_seg_transformed.squeeze(0).long()

        return data_dict