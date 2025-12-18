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

        self.samples: List[dict] = []
        for case_dir in self.cases:
            for required in ["image.png", "gt_1.png", "gt_2.png"]:
                if not os.path.exists(os.path.join(case_dir, required)):
                    raise FileNotFoundError(f"Missing {required} in {case_dir}")

            mask1 = self._load_mask(os.path.join(case_dir, "gt_1.png"))
            component_count = self._count_connected_components(mask1)
            repeats = 3 if component_count > 1 else 1
            for _ in range(repeats):
                self.samples.append({"case_dir": case_dir, "component_count": component_count})

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

    @staticmethod
    def _count_connected_components(mask: np.ndarray) -> int:
        num_labels, _ = cv2.connectedComponents(mask, connectivity=8)
        return max(num_labels - 1, 0)

    @staticmethod
    def _select_single_component(mask: np.ndarray) -> np.ndarray:
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        if num_labels <= 2:
            # A single foreground component or empty mask.
            return mask

        component_labels = np.arange(1, num_labels)
        chosen = np.random.choice(component_labels)
        return (labels == chosen).astype(np.uint8)

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

    def _sample_click(self, mask: np.ndarray, inside: bool) -> Tuple[int, int]:
        """Sample a click biased by distance to the mask centroid.

        When ``inside`` is True, sample from foreground pixels. Otherwise sample
        from background pixels. The selection probability is inversely
        proportional to the distance from the centroid of the foreground mask.
        """

        ys, xs = np.where(mask > 0)
        h, w = mask.shape
        if len(ys) == 0:
            return h // 2, w // 2

        centroid_y = float(ys.mean())
        centroid_x = float(xs.mean())

        if inside:
            candidate_ys, candidate_xs = ys, xs
        else:
            candidate_ys, candidate_xs = np.where(mask == 0)

        if len(candidate_ys) == 0:
            return h // 2, w // 2

        distances = np.sqrt((candidate_ys - centroid_y) ** 2 + (candidate_xs - centroid_x) ** 2)
        weights = 1.0 / (distances + 1e-3)
        probs = weights / weights.sum()
        idx = np.random.choice(len(candidate_ys), p=probs)
        return int(candidate_ys[idx]), int(candidate_xs[idx])

    def __getitem__(self, idx: int):
        sample_info = self.samples[idx]
        case_dir = sample_info["case_dir"]
        component_count = sample_info["component_count"]

        image = self._load_image(case_dir)
        mask1 = self._load_mask(os.path.join(case_dir, "gt_1.png"))
        mask2 = self._load_mask(os.path.join(case_dir, "gt_2.png"))

        click_exists = True
        click_inside_mask = True

        if component_count > 1:
            if np.random.rand() < 0.5:
                mask1 = self._select_single_component(mask1)
                click_inside_mask = True
            else:
                click_exists = False
        else:
            rand_val = np.random.rand()
            if rand_val < 0.7:
                click_inside_mask = True
            elif rand_val < 0.85:
                click_inside_mask = False
            else:
                click_exists = False

        augmented = self.spatial_transform(image=image, mask1=mask1, mask2=mask2)
        image = augmented["image"]
        mask1 = augmented["mask1"]
        mask2 = augmented["mask2"]

        if self.color_transform:
            image = self.color_transform(image=image)["image"]

        h, w = mask1.shape
        if click_exists:
            click_y, click_x = self._sample_click(mask1, inside=click_inside_mask)
            heatmap = self._generate_heatmap(h, w, (click_y, click_x))
        else:
            heatmap = np.zeros((h, w), dtype=np.float32)

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
