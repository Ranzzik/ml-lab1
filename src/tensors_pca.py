from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .io_operations import save_figure, save_table, unique_path
from .preprocessing import MinMaxScalerManual, ZScoreScalerManual


def tensorflow_tensor(frame: pd.DataFrame):
    import tensorflow as tf

    tensor = tf.constant(frame.to_numpy(dtype=np.float32))
    print(f"TensorFlow tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    return tensor


def pytorch_tensor(frame: pd.DataFrame):
    import torch

    tensor = torch.tensor(frame.to_numpy(dtype=np.float32))
    print(f"PyTorch tensor: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    return tensor


def generate_tensor_factors(
    sample_count: int, feature_count: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, columns = 4, 3
    matrix_a = rng.integers(0, 10, size=(rows, sample_count)).astype(np.float32)
    matrix_w = rng.normal(size=(feature_count, columns)).astype(np.float32)
    matrix_b = rng.uniform(-1.0, 1.0, size=(rows, columns)).astype(np.float32)
    return matrix_a, matrix_w, matrix_b


def tensorflow_axw_b(frame: pd.DataFrame, seed: int = 42):
    import tensorflow as tf

    tensor_x = tf.constant(frame.to_numpy(dtype=np.float32))
    matrix_a, matrix_w, matrix_b = generate_tensor_factors(tensor_x.shape[0], tensor_x.shape[1], seed)
    tensor_a = tf.constant(matrix_a)
    tensor_w = tf.constant(matrix_w)
    tensor_b = tf.constant(matrix_b)
    result = tf.linalg.matmul(tf.linalg.matmul(tensor_a, tensor_x), tensor_w) + tensor_b
    print(f"TensorFlow AXW+B: shape={result.shape}, mean={float(tf.reduce_mean(result)):.4f}")
    return result


def pytorch_axw_b(frame: pd.DataFrame, seed: int = 42):
    import torch

    tensor_x = torch.tensor(frame.to_numpy(dtype=np.float32))
    matrix_a, matrix_w, matrix_b = generate_tensor_factors(tensor_x.shape[0], tensor_x.shape[1], seed)
    tensor_a = torch.tensor(matrix_a)
    tensor_w = torch.tensor(matrix_w)
    tensor_b = torch.tensor(matrix_b)
    result = tensor_a @ tensor_x @ tensor_w + tensor_b
    print(f"PyTorch AXW+B: shape={tuple(result.shape)}, mean={float(result.mean()):.4f}")
    return result


def make_tpq(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix_t = rng.normal(size=(3, 10)).astype(np.float32)
    matrix_p = rng.normal(size=(3, 10)).astype(np.float32) * 0.5
    matrix_q = (np.abs(rng.normal(size=(3, 10))) + 0.1).astype(np.float32)
    if np.array_equal(matrix_t, matrix_p) or np.array_equal(matrix_p, matrix_q):
        raise RuntimeError("Generated matrices must be distinct")
    return matrix_t, matrix_p, matrix_q


def keras_expression(matrices: tuple[np.ndarray, np.ndarray, np.ndarray]):
    from keras import ops

    matrix_t, matrix_p, matrix_q = matrices
    tensor_t = ops.convert_to_tensor(matrix_t)
    tensor_p = ops.convert_to_tensor(matrix_p)
    tensor_q = ops.convert_to_tensor(matrix_q)
    result = (ops.sin(tensor_t) - ops.exp(tensor_p)) * tensor_q
    print(f"Keras expression: shape={tuple(result.shape)}, mean={float(ops.mean(result)):.4f}")
    return result


def torch_expression(matrices: tuple[np.ndarray, np.ndarray, np.ndarray]):
    import torch

    matrix_t, matrix_p, matrix_q = matrices
    tensor_t = torch.tensor(matrix_t)
    tensor_p = torch.tensor(matrix_p)
    tensor_q = torch.tensor(matrix_q)
    result = (torch.sin(tensor_t) - torch.exp(tensor_p)) * tensor_q
    print(f"PyTorch expression: shape={tuple(result.shape)}, mean={float(result.mean()):.4f}")
    return result


def sparse_matrix(output_dir: str | Path, seed: int = 42):
    rng = np.random.default_rng(seed)
    row_count, column_count = 100000, 100000
    density = 1e-6
    nonzero_count = int(density * row_count * column_count)
    row_indices = rng.integers(0, row_count, size=nonzero_count, dtype=np.int32)
    column_indices = rng.integers(0, column_count, size=nonzero_count, dtype=np.int32)
    values = rng.random(nonzero_count, dtype=np.float32)
    matrix = sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(row_count, column_count),
    ).tocsr()
    memory_bytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    path = unique_path(output_dir, "46_sparse_matrix", ".npz")
    sparse.save_npz(path, matrix)
    print(
        f"Sparse matrix: shape={matrix.shape}, format={matrix.format}, "
        f"non-zero={matrix.nnz}, memory={memory_bytes / 1024 ** 2:.2f} MiB, saved={path.name}"
    )
    return matrix, path


def random_matrix_inverse(seed: int = 42):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(100, 100))
    determinant = float(np.linalg.det(matrix))
    print(f"Random matrix: shape={matrix.shape}, determinant={determinant:.6e}")
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        print("Random matrix is singular or ill-conditioned; inverse does not exist.")
        return matrix, None, determinant
    inverse = np.linalg.inv(matrix)
    residual = float(np.linalg.norm(matrix @ inverse - np.eye(100)))
    print(f"Inverse matrix exists: residual={residual:.6e}, condition={np.linalg.cond(matrix):.3e}")
    return matrix, inverse, determinant


def build_tensor_f(frame: pd.DataFrame, masks: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    numeric = frame.select_dtypes(include=[np.number])
    original = numeric.to_numpy(dtype=float, copy=False)
    minmax = MinMaxScalerManual((0.0, 1.0)).fit(original).transform(original)
    zscore = ZScoreScalerManual().fit(original).transform(original)
    mask_names = ["positive", "negative", "zero", "interval_-1_1"]
    samples, features = original.shape
    tensor_f = np.empty((samples, 3 + len(mask_names), features), dtype=float)
    tensor_f[:, 0, :] = original
    tensor_f[:, 1, :] = minmax
    tensor_f[:, 2, :] = zscore
    for channel_index, mask_name in enumerate(mask_names, start=3):
        for feature_index, column in enumerate(numeric.columns):
            tensor_f[:, channel_index, feature_index] = masks[column][mask_name]
    view_checks = [
        np.shares_memory(tensor_f, tensor_f[:, 0, :]),
        np.shares_memory(tensor_f, tensor_f[:, 1, :]),
        np.shares_memory(tensor_f, tensor_f[:, 3:, :]),
    ]
    print(f"Tensor F: shape={tensor_f.shape} (samples, channels, features)")
    print(f"NumPy view checks (base with slices): {view_checks}")
    return tensor_f


def _feature_sets(tensor_f: np.ndarray) -> dict[str, np.ndarray]:
    samples = tensor_f.shape[0]
    return {
        "original": tensor_f[:, 0, :],
        "minmax": tensor_f[:, 1, :],
        "zscore": tensor_f[:, 2, :],
        "masks": tensor_f[:, 3:, :].reshape(samples, -1),
        "scaled_plus_masks": tensor_f[:, 1:, :].reshape(samples, -1),
        "all_channels": tensor_f.reshape(samples, -1),
    }


def pca_pipeline_experiments(tensor_f: np.ndarray, results_dir: str | Path):
    feature_sets = _feature_sets(tensor_f)
    scalers = {"standard": StandardScaler(), "minmax": MinMaxScaler()}
    rows = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample_count = tensor_f.shape[0]

    for set_name, dataset in feature_sets.items():
        for scaler_name, scaler in scalers.items():
            max_components = int(min(5, dataset.shape[1], sample_count))
            component_values = []
            log_likelihoods = []
            for components in range(1, max_components + 1):
                pipeline = Pipeline(
                    [
                        ("scaler", clone(scaler)),
                        ("pca", PCA(n_components=components, random_state=42)),
                    ]
                )
                transformed = pipeline.fit_transform(dataset)
                pca = pipeline.named_steps["pca"]
                scaled_input = pipeline.named_steps["scaler"].transform(dataset)
                reconstruction = pca.inverse_transform(transformed)
                mse = float(np.mean((scaled_input - reconstruction) ** 2))
                sigma_squared = max(mse, 1e-12)
                log_likelihood = -0.5 * sample_count * dataset.shape[1] * (
                    np.log(2 * np.pi * sigma_squared) + 1
                )
                explained = pca.explained_variance_ratio_
                rows.append(
                    {
                        "feature_set": set_name,
                        "scaler": scaler_name,
                        "components": components,
                        "log_likelihood": float(log_likelihood),
                        "explained_variance_ratio_sum": float(np.sum(explained)),
                        "projection_quality_2d": float(np.sum(explained[:2])),
                    }
                )
                component_values.append(components)
                log_likelihoods.append(log_likelihood)
            curves[f"{set_name} / {scaler_name}"] = (
                np.asarray(component_values),
                np.asarray(log_likelihoods),
            )

    experiments = pd.DataFrame(rows)
    two_component_rows = experiments[experiments["components"] == 2]
    if two_component_rows.empty:
        best_row = experiments.loc[experiments["log_likelihood"].idxmax()]
    else:
        best_row = two_component_rows.loc[two_component_rows["log_likelihood"].idxmax()]
    best_set = str(best_row["feature_set"])
    best_scaler = str(best_row["scaler"])

    fig, ax = plt.subplots(figsize=(12, 7))
    for label, (components, log_likelihoods) in curves.items():
        ax.plot(components, log_likelihoods, marker="o", linewidth=1.4, label=label)
    ax.set_title("PCA pipeline log-likelihood versus number of components")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Log-likelihood")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    likelihood_path = save_figure(fig, results_dir, "49_pca_log_likelihood")
    plt.close(fig)

    best_dataset = feature_sets[best_set]
    best_pipeline = Pipeline(
        [
            ("scaler", clone(scalers[best_scaler])),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )
    projection = best_pipeline.fit_transform(best_dataset)
    fig, ax = plt.subplots(figsize=(8, 6))
    color = tensor_f[:, 0, 0]
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=color, cmap="viridis", s=28, alpha=0.8, edgecolors="k", linewidths=0.3)
    fig.colorbar(scatter, ax=ax, label="Original feature 1")
    ax.set_title(f"Best 2D PCA projection: {best_set} / {best_scaler}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    projection_path = save_figure(fig, results_dir, "49_best_pca_projection")
    plt.close(fig)

    print(
        f"Best PCA projection: feature_set={best_set}, scaler={best_scaler}, "
        f"log_likelihood={best_row['log_likelihood']:.4f}, "
        f"2D explained variance={best_row['projection_quality_2d']:.4f}"
    )
    return experiments, {
        "feature_set": best_set,
        "scaler": best_scaler,
        "components": int(best_row["components"]),
        "log_likelihood": float(best_row["log_likelihood"]),
        "projection_quality_2d": float(best_row["projection_quality_2d"]),
    }, likelihood_path, projection_path
