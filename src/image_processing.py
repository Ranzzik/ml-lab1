from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA

from .io_operations import save_figure, unique_path


def ensure_test_image(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 320, 320
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    red = (x_grid / (width - 1) * 255).astype(np.float64)
    green = (y_grid / (height - 1) * 255).astype(np.float64)
    blue = (0.5 + 0.5 * np.sin(x_grid / 22.0) * np.cos(y_grid / 27.0)) * 255
    circle = ((x_grid - width * 0.35) ** 2 + (y_grid - height * 0.45) ** 2) < (width * 0.18) ** 2
    red[circle] = 245
    green[circle] = 210
    blue[circle] = 40
    square = (x_grid > width * 0.62) & (x_grid < width * 0.88) & (y_grid > height * 0.60) & (y_grid < height * 0.85)
    red[square] = 30
    green[square] = 170
    blue[square] = 220
    image = np.stack([red, green, blue], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)
    return path


def _save_array_image(array: np.ndarray, results_dir: str | Path, basename: str) -> Path:
    path = unique_path(results_dir, basename, ".png")
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        Image.fromarray(array, mode="L").save(path)
    else:
        Image.fromarray(array, mode="RGB").save(path)
    return path


def _thresholds_from_histogram(gray: np.ndarray) -> tuple[float, float]:
    histogram, edges = np.histogram(gray, bins=256, range=(0, 255))
    cumulative = np.cumsum(histogram)
    cumulative = cumulative / max(cumulative[-1], 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    first_threshold = float(centers[np.searchsorted(cumulative, 1 / 3)])
    second_threshold = float(centers[np.searchsorted(cumulative, 2 / 3)])
    if first_threshold == second_threshold:
        second_threshold = first_threshold + 1.0
    return first_threshold, second_threshold


def run_image_pipeline(image_path: str | Path, results_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    image_path = ensure_test_image(image_path)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64)
    paths: dict[str, Path] = {}
    rng = np.random.default_rng(seed)

    channels = [image[:, :, index] for index in range(3)]
    for name, channel in zip(["red", "green", "blue"], channels):
        paths[f"channel_{name}"] = _save_array_image(channel, results_dir, f"50_channel_{name}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name, channel in zip(axes, ["Red", "Green", "Blue"], channels):
        ax.imshow(channel, cmap="gray")
        ax.set_title(f"{name} channel")
        ax.axis("off")
    fig.tight_layout()
    paths["channels_figure"] = save_figure(fig, results_dir, "50_color_channels")
    plt.close(fig)

    rotated = np.stack(
        [ndimage.rotate(channel, 40, reshape=False, mode="constant", cval=0.0) for channel in channels],
        axis=-1,
    )
    paths["rotated"] = _save_array_image(rotated, results_dir, "50_rotated_40_degrees")

    gray = np.dot(image, np.array([0.299, 0.587, 0.114]))
    paths["grayscale"] = _save_array_image(gray, results_dir, "50_grayscale")

    histogram, edges = np.histogram(gray, bins=256, range=(0, 255))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(edges[:-1], histogram, width=np.diff(edges), color="dimgray", align="edge")
    ax.set_title("Grayscale image histogram")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    paths["histogram"] = save_figure(fig, results_dir, "50_grayscale_histogram")
    plt.close(fig)

    lower, upper = _thresholds_from_histogram(gray)
    digitized = np.digitize(gray, [lower, upper])
    split_images = []
    for index in range(3):
        mask = digitized == index
        split = np.zeros_like(image)
        split[mask] = image[mask]
        split_images.append(split)
        paths[f"split_{index + 1}"] = _save_array_image(split, results_dir, f"50_histogram_part_{index + 1}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, index, split in zip(axes, range(3), split_images):
        ax.imshow(split.astype(np.uint8))
        ax.set_title(f"Histogram range {index + 1}")
        ax.axis("off")
    fig.tight_layout()
    paths["split_figure"] = save_figure(fig, results_dir, "50_histogram_split")
    plt.close(fig)

    height, width = gray.shape
    y_grid, x_grid = np.indices((height, width))
    radius = 0.48 * min(height, width)
    circular_mask = (x_grid - width / 2.0) ** 2 + (y_grid - height / 2.0) ** 2 <= radius ** 2
    framed = image * circular_mask[:, :, np.newaxis]
    paths["circular_frame"] = _save_array_image(framed, results_dir, "50_circular_frame")

    noise = rng.normal(0.0, 18.0, size=image.shape)
    noisy = np.clip(image + noise, 0, 255)
    paths["noisy"] = _save_array_image(noisy, results_dir, "50_gaussian_noise")

    blurred = ndimage.gaussian_filter(image, sigma=(2.0, 2.0, 0.0))
    paths["blurred"] = _save_array_image(blurred, results_dir, "50_gaussian_blur")

    soft = ndimage.gaussian_filter(image, sigma=(1.0, 1.0, 0.0))
    sharpened = np.clip(image + 1.6 * (image - soft), 0, 255)
    paths["sharpened"] = _save_array_image(sharpened, results_dir, "50_sharpened")

    pixels = image.reshape(-1, 3)
    pca = PCA(n_components=3, random_state=seed)
    scores = pca.fit_transform(pixels)
    reconstructed = pca.inverse_transform(scores).reshape(image.shape)
    paths["pca_reconstruction"] = _save_array_image(reconstructed, results_dir, "50_pca_reconstruction")

    sample_size = min(6000, len(pixels))
    sample_indices = rng.choice(len(pixels), size=sample_size, replace=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        scores[sample_indices, 0],
        scores[sample_indices, 1],
        c=np.clip(pixels[sample_indices] / 255.0, 0, 1),
        s=8,
        alpha=0.7,
    )
    ax.set_title("PCA projection of RGB pixels")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    paths["pca_scatter"] = save_figure(fig, results_dir, "50_pca_rgb_scatter")
    plt.close(fig)

    print(f"Image processing: {len(paths)} outputs, histogram thresholds=({lower:.1f}, {upper:.1f})")
    return paths
