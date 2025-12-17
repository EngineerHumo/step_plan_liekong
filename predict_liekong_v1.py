import argparse
import os
from pathlib import Path
from typing import List, Tuple

# 1. 调整 import 顺序，确保在 import pyplot 之前配置好 backend（可选，但推荐）
import matplotlib
# 如果你的环境装了 PyQt5，可以用 'Qt5Agg'，否则用 'TkAgg' (Python自带)
# 如果不确定，可以把下面这行注释掉，让 matplotlib 自动选择
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from model import PRPSegmenter


# ================= 修改处 1：删除导致报错的代码 =================
# 原代码：plt.switch_backend("Agg") if os.environ.get("DISPLAY", "") == "" else None
# 原因：Windows下没有 DISPLAY 环境变量，这行代码强制关闭了显示窗口，导致 ginput 无法工作。
# ==============================================================

def load_model(model_path: str, device: torch.device) -> PRPSegmenter:
    model = PRPSegmenter(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def generate_gaussian_heatmap(height: int, width: int, center: Tuple[float, float], sigma: float) -> np.ndarray:
    y = np.arange(0, height, 1, float)
    x = np.arange(0, width, 1, float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    heatmap = np.exp(-((yy - center[1]) ** 2 + (xx - center[0]) ** 2) / (2 * sigma ** 2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.astype(np.float32)


def binary_dilation_keep_largest(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    grid = np.arange(-radius, radius + 1)
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    kernel = (gx ** 2 + gy ** 2) <= radius ** 2
    dilated = ndi.binary_dilation(mask, structure=kernel)
    labeled, num = ndi.label(dilated)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(mask, dtype=np.int32), labels=labeled, index=range(1, num + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    return labeled == largest_idx


def remove_overlap(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return mask & (~reference)


def greedy_circle_centers(
    band_mask: np.ndarray,
    min_center_dist: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    ys, xs = np.where(band_mask)
    # 为了避免按扫描顺序导致的局部堆积，随机打乱候选点顺序
    order = rng.permutation(len(xs))
    centers: List[Tuple[int, int]] = []
    for idx in order:
        y, x = ys[idx], xs[idx]
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_center_dist ** 2 for cy, cx in centers):
            centers.append((y, x))
    return centers


def plan_circle_layout(
    gt1_mask: np.ndarray,
    gt2_mask: np.ndarray,
    radius: int,
    spacing: int,
) -> List[Tuple[int, int]]:
    """
    沿着 gt_2 中与 gt_1 相邻的边界，在 gt_2 内布置三圈蓝色圆形。

    逻辑：
    1. 找到 gt_2 与 gt_1 相邻的边界像素（在 gt_2 内，且 8 邻域接触 gt_1）。
    2. 以该边界作为参考，计算 gt_2 内到边界的距离，并按距离分三圈取样。
    3. 蓝色圆之间的最小间距基于圆边缘，因此圆心间距 = 直径 + 最小间距。
    4. 仅在 gt_2 内放置圆形，不在 gt_1 内放置。
    """

    if gt2_mask.sum() == 0 or gt1_mask.sum() == 0:
        return []

    # 仅考虑 gt_2 内与 gt_1 相邻的边界区域
    adjacency = gt2_mask & ndi.binary_dilation(gt1_mask, structure=np.ones((3, 3)))
    if adjacency.sum() == 0:
        return []

    # 计算到接触边界的距离（仅在 gt_2 内有效），并限制最大距离确保只有三圈
    dist = ndi.distance_transform_edt(~adjacency) * gt2_mask
    effective_spacing = spacing + 2 * radius  # 圆心间距，保证圆边界间距为 spacing
    max_distance = radius + 2 * effective_spacing + spacing  # 超过三圈的距离范围全部舍弃

    centers: List[Tuple[int, int]] = []
    rng = np.random.default_rng(42)
    band_half_width = max(1, effective_spacing // 3)

    for ring_idx in range(3):
        target = radius + ring_idx * effective_spacing
        band = (
            (dist >= target - band_half_width)
            & (dist <= target + band_half_width)
            & (dist <= max_distance)
            & gt2_mask
        )
        if band.sum() == 0:
            continue
        new_centers = greedy_circle_centers(band, effective_spacing, rng)
        centers.extend(new_centers)
    return centers


def draw_circles_on_image(base_image: Image.Image, centers: List[Tuple[int, int]], diameter: int) -> Image.Image:
    canvas = base_image.copy()
    draw = ImageDraw.Draw(canvas)
    radius = diameter // 2
    for y, x in centers:
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline="blue", width=2)
    return canvas


def infer_with_click(
        model: PRPSegmenter,
        image: Image.Image,
        click_xy: Tuple[float, float],
        sigma: float,
        device: torch.device,
        gt1_threshold: float,
        gt2_threshold: float,
        circle_diameter: int,
        circle_spacing: int,
) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    input_image = image.resize((1280, 1280), Image.BILINEAR)
    # 注意：这里的 1240 可能是原代码的一个硬编码尺寸，确保和你的模型训练尺寸一致
    click_scaled = (click_xy[0] / 1240 * 1280, click_xy[1] / 1240 * 1280)
    heatmap_np = generate_gaussian_heatmap(1280, 1280, click_scaled, sigma)

    image_tensor = pil_to_tensor(input_image).unsqueeze(0).to(device)
    heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred1 = model(image_tensor, heatmap_tensor)

    pred1_resized = F.interpolate(pred1, size=(1240, 1240), mode="bilinear", align_corners=False)
    gt1 = (pred1_resized.squeeze().cpu().numpy() >= gt1_threshold)

    gt1_processed = binary_dilation_keep_largest(gt1, radius=12)
    gt2_processed = np.zeros_like(gt1_processed, dtype=bool)

    centers = plan_circle_layout(
        gt1_mask=gt1_processed,
        gt2_mask=gt2_processed,
        radius=circle_diameter // 2,
        spacing=circle_spacing,
    )
    resized_image = image.resize((1240, 1240), Image.BILINEAR)
    overlay = draw_circles_on_image(resized_image, centers, diameter=circle_diameter)

    return gt1_processed.astype(np.uint8), gt2_processed.astype(np.uint8), overlay


def display_intermediate(image: Image.Image) -> Tuple[float, float]:
    # ================= 修改处 2：优化显示逻辑 =================
    print("正在打开图像窗口，请在病灶位置点击鼠标左键...")

    # 启用交互模式，确保窗口不会阻塞导致假死（虽然 ginput 本身是阻塞的）
    plt.ion()
    fig = plt.figure("Input Image", figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("单击选择提示位置 (Click to select prompt)")

    # 强制绘制一下，防止窗口空白
    plt.draw()
    plt.pause(0.1)

    # 获取点击输入，timeout=0 表示无限等待直到点击
    coords = plt.ginput(1, timeout=0, mouse_add=1, mouse_stop=None, mouse_pop=None)

    plt.close(fig)
    plt.ioff()  # 关闭交互模式
    # =======================================================

    if not coords:
        raise RuntimeError("未检测到点击，或者窗口被直接关闭。请重新运行并点击图像。")

    print(f"捕获点击坐标: {coords[0]}")
    return coords[0]


def visualize_results(gt1: np.ndarray, gt2: np.ndarray, overlay: Image.Image):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt1, cmap="gray")
    axes[0].set_title("gt_1 后处理")
    axes[1].imshow(gt2, cmap="gray")
    axes[1].set_title("gt_2 去重叠")
    axes[2].imshow(overlay)
    axes[2].set_title("gt_2 蓝色圆形标注")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def save_outputs(gt1: np.ndarray, gt2: np.ndarray, overlay: Image.Image, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    gt1_img = Image.fromarray(gt1 * 255)
    gt2_img = Image.fromarray(gt2 * 255)
    gt1_path = output_dir / f"{stem}_gt_1.png"
    gt2_path = output_dir / f"{stem}_gt_2.png"
    overlay_path = output_dir / f"{stem}_gt_2_overlay.png"
    gt1_img.save(gt1_path)
    gt2_img.save(gt2_path)
    overlay.save(overlay_path)
    print(f"保存 gt_1 至 {gt1_path}")
    print(f"保存 gt_2 至 {gt2_path}")
    print(f"保存带圆形标注的 gt_2 至 {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="交互式点击预测并生成后处理分割结果")
    # 请确保这里的路径是你本地实际存在的路径
    parser.add_argument("--model-path", default=r"C:\work space\liekoong\predict\best_model.pth", help="模型权重路径")
    parser.add_argument("--image-path", default=r"C:\work space\liekoong\demo\20250529110843581.bmp",
                        help="待预测图像路径")
    parser.add_argument("--output-dir", default="outputs", help="输出保存目录")
    parser.add_argument("--device", default=None, help="使用的设备，如 cuda:0 或 cpu")
    parser.add_argument("--gt1-threshold", type=float, default=0.5, help="gt_1 阈值")
    parser.add_argument("--gt2-threshold", type=float, default=0.4, help="gt_2 阈值")
    parser.add_argument("--sigma", type=float, default=15.0, help="点击生成高斯热图的标准差")
    parser.add_argument("--circle-diameter", type=int, default=15, help="绘制蓝色圆的直径")
    parser.add_argument(
        "--circle-spacing",
        type=int,
        default=10,
        help="蓝色圆之间的最小边界间距（像素）",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 简单的文件存在性检查
    if not os.path.exists(args.model_path):
        print(f"Error: 模型文件不存在: {args.model_path}")
        return
    if not os.path.exists(args.image_path):
        print(f"Error: 图像文件不存在: {args.image_path}")
        return

    model = load_model(args.model_path, device)

    image = Image.open(args.image_path).convert("RGB")
    display_image = image.resize((1240, 1240), Image.BILINEAR)

    # 这一步会弹出窗口等待点击
    click_xy = display_intermediate(display_image)

    gt1, gt2, overlay = infer_with_click(
        model=model,
        image=image,
        click_xy=click_xy,
        sigma=args.sigma,
        device=device,
        gt1_threshold=args.gt1_threshold,
        gt2_threshold=args.gt2_threshold,
        circle_diameter=args.circle_diameter,
        circle_spacing=args.circle_spacing,
    )

    visualize_results(gt1, gt2, overlay)

    output_dir = Path(args.output_dir)
    stem = Path(args.image_path).stem
    save_outputs(gt1, gt2, overlay, output_dir, stem)


if __name__ == "__main__":
    main()
