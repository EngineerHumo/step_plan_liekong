import os
from glob import glob
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform


class PRPDataset(torch.utils.data.Dataset):
    """Dataset for semi-automatic PRP segmentation with dual targets.

    Each case folder must contain:
        - image.png
        - gt_1.png
        - gt_2.png

    A simulated click heatmap is generated from gt_1 and all images/masks are
    resized to ``target_size`` (default: 1280x1280). Spatial augmentations are
    applied consistently across the image, both masks, and the heatmap to keep
    alignment intact.
    """

    def __init__(
        self,
        root_dir: str,
        image_extensions: Optional[List[str]] = None,
        augment: bool = True,
        target_size: Tuple[int, int] = (1280, 1280),
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.augment = augment
        self.image_extensions = image_extensions or [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        self.target_size = target_size

        self.cases = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
        if not self.cases:
            raise ValueError(f"No case folders found in {root_dir}")

        self.samples: List[str] = []
        for case_dir in self.cases:
            for required in ["image.png", "gt_1.png", "gt_2.png"]:
                if not os.path.exists(os.path.join(case_dir, required)):
                    raise FileNotFoundError(f"Missing {required} in {case_dir}")
            self.samples.append(case_dir)

        self.spatial_transform = self._build_spatial_transform()
        self.color_transform = self._build_color_transform() if augment else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def _build_spatial_transform(self) -> A.Compose:
        transforms: list[BasicTransform] = []
        if self.augment:
            transforms.extend(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=10,
                        p=0.7,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0,
                    ),
                    A.HorizontalFlip(p=0.5),
                ]
            )

        transforms.append(
            A.Resize(
                height=self.target_size[0],
                width=self.target_size[1],
                interpolation=cv2.INTER_LINEAR,
            )
        )

        return A.Compose(
            transforms,
            additional_targets={
                "mask1": "mask",
                "mask2": "mask",
            },
        )

    def _build_color_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=8, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            ]
        )

    def _load_image(self, case_dir: str) -> np.ndarray:
        image_path = os.path.join(case_dir, "image.png")
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                return image

        # Fallback: try other known extensions but strictly prefer files named "image.*"
        for ext in self.image_extensions:
            candidate = os.path.join(case_dir, f"image{ext}")
            if os.path.exists(candidate):
                image = cv2.imread(candidate)
                if image is not None:
                    return image

        raise FileNotFoundError(
            f"No image found in {case_dir}. Expected image.png or image with extensions {self.image_extensions}"
        )

    def _load_mask(self, mask_path: str) -> np.ndarray:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        return mask_bin.astype(np.uint8)

    @staticmethod
    def _generate_heatmap(height: int, width: int, center: Tuple[int, int], sigma: float = 15.0) -> np.ndarray:
        y = np.arange(0, height, 1, float)
        x = np.arange(0, width, 1, float)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        heatmap = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma ** 2))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.astype(np.float32)

    def _sample_click(self, mask: np.ndarray) -> Tuple[int, int]:
        """Sample a click biased by distance to the mask centroid.

        80% probability to sample inside the mask and 20% outside (within 320
        pixels). The selection probability is inversely proportional to the
        distance to the centroid of ``mask``.
        """

        ys, xs = np.where(mask > 0)
        h, w = mask.shape
        if len(ys) == 0:
            return h // 2, w // 2

        centroid_y = float(ys.mean())
        centroid_x = float(xs.mean())

        inside = np.random.rand() < 0.8

        if inside:
            distances = np.sqrt((ys - centroid_y) ** 2 + (xs - centroid_x) ** 2)
            weights = 1.0 / (distances + 1e-3)
            probs = weights / weights.sum()
            idx = np.random.choice(len(ys), p=probs)
            return int(ys[idx]), int(xs[idx])

        # Outside sampling within 320 px radius
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        distances = np.sqrt((grid_y - centroid_y) ** 2 + (grid_x - centroid_x) ** 2)
        outside_mask = (mask == 0) & (distances <= 320)
        candidate_ys, candidate_xs = np.where(outside_mask)
        if len(candidate_ys) == 0:
            return h // 2, w // 2

        outside_distances = distances[outside_mask]
        weights = 1.0 / (outside_distances + 1e-3)
        probs = weights / weights.sum()
        idx = np.random.choice(len(candidate_ys), p=probs)
        return int(candidate_ys[idx]), int(candidate_xs[idx])

    def __getitem__(self, idx: int):
        case_dir = self.samples[idx]
        image = self._load_image(case_dir)
        mask1 = self._load_mask(os.path.join(case_dir, "gt_1.png"))
        mask2 = self._load_mask(os.path.join(case_dir, "gt_2.png"))

        augmented = self.spatial_transform(image=image, mask1=mask1, mask2=mask2)
        image = augmented["image"]
        mask1 = augmented["mask1"]
        mask2 = augmented["mask2"]

        if self.color_transform:
            image = self.color_transform(image=image)["image"]

        h, w = mask1.shape
        click_y, click_x = self._sample_click(mask1)
        heatmap = self._generate_heatmap(h, w, (click_y, click_x))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        tensor_transform = A.Compose(
            [ToTensorV2()],
            additional_targets={
                "mask1": "mask",
                "mask2": "mask",
                "heatmap": "mask",
            },
        )
        tensors = tensor_transform(image=image, mask1=mask1, mask2=mask2, heatmap=heatmap)
        image_tensor = tensors["image"]
        mask1_tensor = tensors["mask1"].unsqueeze(0).float()
        mask2_tensor = tensors["mask2"].unsqueeze(0).float()
        heatmap_tensor = tensors["heatmap"].float().unsqueeze(0)

        return image_tensor, heatmap_tensor, mask1_tensor, mask2_tensor
