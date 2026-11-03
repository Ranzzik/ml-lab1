from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate

from .io_operations import save_figure


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.select_dtypes(include=[np.number])


def plot_raw_data(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    fig, axes = plt.subplots(len(numeric.columns), 1, figsize=(11, 3.2 * len(numeric.columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], numeric.columns):
        ax.plot(np.arange(len(numeric)), numeric[column].to_numpy(), linewidth=1.4)
        ax.set_title(f"Raw data: {column}")
        ax.set_xlabel("Index")
        ax.set_ylabel(column)
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "14_raw_data")
    plt.close(fig)
    return path


def plot_histograms(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    fig, axes = plt.subplots(1, len(numeric.columns), figsize=(6 * len(numeric.columns), 4.5), squeeze=False)
    for ax, column in zip(axes[0], numeric.columns):
        values = numeric[column].dropna().to_numpy()
        ax.hist(values, bins=30, density=True, alpha=0.75, color="steelblue", edgecolor="black")
        ax.set_title(f"Normalized histogram: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "15_histograms")
    plt.close(fig)
    return path


def sorted_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    numeric = _numeric(frame)
    result = {column: np.sort(numeric[column].dropna().to_numpy()) for column in numeric.columns}
    for column, values in result.items():
        print(f"Sorted {column}: first={values[:5]}, last={values[-5:]}")
    return result


def plot_ecdf(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    fig, axes = plt.subplots(1, len(numeric.columns), figsize=(6 * len(numeric.columns), 4.5), squeeze=False)
    for ax, column in zip(axes[0], numeric.columns):
        values = np.sort(numeric[column].dropna().to_numpy())
        probabilities = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, probabilities, where="post", color="darkorange")
        ax.set_title(f"Empirical CDF: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("F(x)")
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "17_ecdf")
    plt.close(fig)
    return path


def compute_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric(frame)
    rows = []
    for column in numeric.columns:
        values = numeric[column].dropna().to_numpy()
        rows.append(
            {
                "column": column,
                "numpy_mean": float(np.mean(values)),
                "pandas_mean": float(numeric[column].mean()),
                "numpy_variance": float(np.var(values, ddof=1)),
                "pandas_variance": float(numeric[column].var(ddof=1)),
                "median": float(np.median(values)),
                "mode": float(numeric[column].mode().iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_statistics(statistics: pd.DataFrame, results_dir: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(statistics))
    width = 0.35
    axes[0].bar(x - width / 2, statistics["numpy_mean"], width, label="Mean")
    axes[0].bar(x + width / 2, statistics["median"], width, label="Median")
    axes[0].set_xticks(x, statistics["column"])
    axes[0].set_title("Mean and median")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].bar(x - width / 2, statistics["numpy_variance"], width, label="Variance")
    axes[1].bar(x + width / 2, statistics["mode"], width, label="Mode")
    axes[1].set_xticks(x, statistics["column"])
    axes[1].set_title("Variance and mode")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "18_statistics")
    plt.close(fig)
    return path


def confidence_intervals(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    numeric = _numeric(frame)
    rows = []
    for column in numeric.columns:
        values = numeric[column].dropna().to_numpy()
        sample_size = len(values)
        mean = float(np.mean(values))
        variance = float(np.var(values, ddof=1))
        standard_error = np.sqrt(variance / sample_size)
        t_critical = float(stats.t.ppf(1 - alpha / 2, df=sample_size - 1))
        lower_mean = mean - t_critical * standard_error
        upper_mean = mean + t_critical * standard_error
        chi2_lower = float(stats.chi2.ppf(alpha / 2, df=sample_size - 1))
        chi2_upper = float(stats.chi2.ppf(1 - alpha / 2, df=sample_size - 1))
        lower_variance = (sample_size - 1) * variance / chi2_upper
        upper_variance = (sample_size - 1) * variance / chi2_lower
        rows.append(
            {
                "column": column,
                "mean": mean,
                "mean_ci_lower": lower_mean,
                "mean_ci_upper": upper_mean,
                "variance": variance,
                "variance_ci_lower": lower_variance,
                "variance_ci_upper": upper_variance,
            }
        )
    return pd.DataFrame(rows)


def plot_confidence_intervals(intervals: pd.DataFrame, results_dir: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(intervals))
    mean_error = np.vstack(
        [
            intervals["mean"] - intervals["mean_ci_lower"],
            intervals["mean_ci_upper"] - intervals["mean"],
        ]
    )
    variance_error = np.vstack(
        [
            intervals["variance"] - intervals["variance_ci_lower"],
            intervals["variance_ci_upper"] - intervals["variance"],
        ]
    )
    axes[0].errorbar(x, intervals["mean"], yerr=mean_error, fmt="o", capsize=5)
    axes[0].set_xticks(x, intervals["column"])
    axes[0].set_title("Confidence interval for mean (95%)")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[1].errorbar(x, intervals["variance"], yerr=variance_error, fmt="s", capsize=5, color="darkorange")
    axes[1].set_xticks(x, intervals["column"])
    axes[1].set_title("Confidence interval for variance (95%)")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "19_confidence_intervals")
    plt.close(fig)
    return path


def covariance_correlation(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    covariance = numeric.cov()
    correlation = numeric.corr()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in [
        (axes[0], covariance, "Covariance"),
        (axes[1], correlation, "Correlation"),
    ]:
        image = ax.imshow(matrix.to_numpy(), cmap="coolwarm")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.columns)), matrix.columns)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "20_covariance_correlation")
    plt.close(fig)
    return covariance, correlation, path


def correlation_significance(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric(frame)
    columns = list(numeric.columns)
    rows = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            first = numeric[columns[i]].dropna().to_numpy()
            second = numeric[columns[j]].dropna().to_numpy()
            size = min(len(first), len(second))
            coefficient, p_value = stats.pearsonr(first[:size], second[:size])
            rows.append(
                {
                    "pair": f"{columns[i]}-{columns[j]}",
                    "pearson_r": float(coefficient),
                    "p_value": float(p_value),
                    "significant_alpha_0.05": bool(p_value < 0.05),
                }
            )
    result = pd.DataFrame(rows)
    print("\nCorrelation significance:")
    print(result.to_string(index=False))
    return result


def cross_correlation(frame: pd.DataFrame, max_lags: int = 30, results_dir: str | Path | None = None):
    numeric = _numeric(frame)
    columns = list(numeric.columns)
    fig, axes = plt.subplots(len(columns), len(columns), figsize=(5 * len(columns), 4 * len(columns)), squeeze=False)
    results = {}
    for row_index, first_column in enumerate(columns):
        for column_index, second_column in enumerate(columns):
            first = numeric[first_column].dropna().to_numpy()
            second = numeric[second_column].dropna().to_numpy()
            size = min(len(first), len(second))
            full = correlate(first[:size], second[:size], mode="full")
            lags = np.arange(-size + 1, size)
            mask = np.abs(lags) <= max_lags
            normalized = full / max(np.max(np.abs(full)), 1e-12)
            ax = axes[row_index, column_index]
            ax.plot(lags[mask], normalized[mask], linewidth=1.3)
            ax.set_title(f"{first_column} vs {second_column}")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Normalized correlation")
            ax.grid(True, linestyle=":", alpha=0.5)
            results[(first_column, second_column)] = (lags[mask], normalized[mask])
    fig.tight_layout()
    path = save_figure(fig, results_dir, "22_cross_correlation") if results_dir is not None else None
    plt.close(fig)
    return results, path


def numerical_derivative_gradient(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    values = numeric.to_numpy(dtype=float)
    gradient = np.gradient(values, axis=0, edge_order=2)
    derivatives = np.gradient(values, axis=0, edge_order=1)
    fig, axes = plt.subplots(len(numeric.columns), 1, figsize=(11, 3.2 * len(numeric.columns)), squeeze=False)
    for index, (ax, column) in enumerate(zip(axes[:, 0], numeric.columns)):
        ax.plot(np.arange(len(values)), derivatives[:, index], label="Numerical derivative", linewidth=1.3)
        ax.plot(np.arange(len(values)), gradient[:, index], label="Gradient", linestyle="--", linewidth=1.3)
        ax.set_title(f"Derivative and gradient: {column}")
        ax.set_xlabel("Index")
        ax.set_ylabel("d/dindex")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "23_derivatives_gradients")
    plt.close(fig)
    return derivatives, gradient, path


def convolution(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    columns = list(numeric.columns)
    results = {}
    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 3.2 * len(columns)), squeeze=False)
    for index, (ax, column) in enumerate(zip(axes[:, 0], columns)):
        values = numeric[column].dropna().to_numpy()
        convolved = np.convolve(values, values, mode="full")
        results[(column, column)] = convolved
        ax.plot(convolved, linewidth=1.2)
        ax.set_title(f"Self-convolution: {column}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=":", alpha=0.5)
    if len(columns) >= 2:
        first = numeric[columns[0]].dropna().to_numpy()
        second = numeric[columns[1]].dropna().to_numpy()
        size = min(len(first), len(second))
        cross = np.convolve(first[:size], second[:size], mode="full")
        results[(columns[0], columns[1])] = cross
        print(f"Cross-convolution {columns[0]} * {columns[1]}: length={len(cross)}, max={cross.max():.4f}")
    fig.tight_layout()
    path = save_figure(fig, results_dir, "24_convolution")
    plt.close(fig)
    return results, path


def dot_and_cross_products(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = _numeric(frame)
    columns = list(numeric.columns)
    dot_rows = []
    cross_rows = []
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            first = numeric[columns[i]].dropna().to_numpy()
            second = numeric[columns[j]].dropna().to_numpy()
            size = min(len(first), len(second))
            dot_rows.append(
                {
                    "pair": f"{columns[i]}-{columns[j]}",
                    "scalar_product": float(np.dot(first[:size], second[:size])),
                }
            )
            cutoff = size - (size % 3)
            if cutoff >= 3:
                first_vectors = first[:cutoff].reshape(-1, 3)
                second_vectors = second[:cutoff].reshape(-1, 3)
                cross = np.cross(first_vectors, second_vectors)
                cross_rows.append(
                    {
                        "pair": f"{columns[i]}-{columns[j]}",
                        "vector_product_shape": cross.shape,
                        "vector_product_norm": float(np.linalg.norm(cross)),
                    }
                )
    dot_frame = pd.DataFrame(dot_rows)
    cross_frame = pd.DataFrame(cross_rows)
    print("\nScalar products:")
    print(dot_frame.to_string(index=False))
    print("\nVector products (columns reshaped to 3D vectors):")
    print(cross_frame.to_string(index=False))
    return dot_frame, cross_frame


def column_norms(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric(frame)
    rows = []
    for column in numeric.columns:
        values = numeric[column].dropna().to_numpy()
        rows.append(
            {
                "column": column,
                "L1_norm": float(np.linalg.norm(values, ord=1)),
                "L2_norm": float(np.linalg.norm(values, ord=2)),
            }
        )
    result = pd.DataFrame(rows)
    print("\nColumn norms:")
    print(result.to_string(index=False))
    return result


def distribution_tests(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = _numeric(frame)
    columns = list(numeric.columns)
    rows = []
    if columns:
        values = numeric[columns[0]].dropna().to_numpy()
        low, high = float(np.min(values)), float(np.max(values))
        statistic, p_value = stats.kstest(values, "uniform", args=(low, high - low))
        rows.append(
            {
                "column": columns[0],
                "test": "uniform (Kolmogorov-Smirnov)",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "reject_H0_alpha_0.05": bool(p_value < 0.05),
            }
        )
    if len(columns) >= 2:
        values = numeric[columns[1]].dropna().to_numpy()
        statistic, p_value = stats.normaltest(values)
        rows.append(
            {
                "column": columns[1],
                "test": "normal (D'Agostino-Pearson)",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "reject_H0_alpha_0.05": bool(p_value < 0.05),
            }
        )
    result = pd.DataFrame(rows)
    print("\nDistribution tests:")
    print(result.to_string(index=False))
    return result


def scaling_comparison(frame: pd.DataFrame, results_dir: str | Path) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    from .preprocessing import MinMaxScalerManual, ZScoreScalerManual

    numeric = _numeric(frame)
    columns = list(numeric.columns)
    rows = []
    fig, axes = plt.subplots(2, len(columns), figsize=(6.5 * len(columns), 8), squeeze=False)
    for index, column in enumerate(columns):
        values = numeric[column].dropna().to_numpy()
        reshaped = values.reshape(-1, 1)

        manual_minmax = MinMaxScalerManual((0.0, 1.0)).fit(values)
        manual_minmax_values = manual_minmax.transform(values)
        sklearn_minmax = MinMaxScaler().fit(reshaped)
        sklearn_minmax_values = sklearn_minmax.transform(reshaped).ravel()

        manual_zscore = ZScoreScalerManual().fit(values)
        manual_zscore_values = manual_zscore.transform(values)
        sklearn_zscore = StandardScaler().fit(reshaped)
        sklearn_zscore_values = sklearn_zscore.transform(reshaped).ravel()

        restored_manual_minmax = manual_minmax.inverse_transform(manual_minmax_values)
        restored_sklearn_minmax = sklearn_minmax.inverse_transform(sklearn_minmax_values.reshape(-1, 1)).ravel()
        restored_manual_zscore = manual_zscore.inverse_transform(manual_zscore_values)
        restored_sklearn_zscore = sklearn_zscore.inverse_transform(sklearn_zscore_values.reshape(-1, 1)).ravel()

        axes[0, index].plot(values, manual_minmax_values, label="Manual MinMax", linewidth=1.4)
        axes[0, index].plot(values, sklearn_minmax_values, label="sklearn MinMax", linestyle="--", linewidth=1.4)
        axes[0, index].set_title(f"MinMax comparison: {column}")
        axes[0, index].set_xlabel(column)
        axes[0, index].set_ylabel("Scaled value")
        axes[0, index].legend(fontsize=8)
        axes[0, index].grid(True, linestyle=":", alpha=0.5)

        axes[1, index].plot(values, manual_zscore_values, label="Manual Z-score", linewidth=1.4)
        axes[1, index].plot(values, sklearn_zscore_values, label="sklearn Standard", linestyle="--", linewidth=1.4)
        axes[1, index].set_title(f"Z-score comparison: {column}")
        axes[1, index].set_xlabel(column)
        axes[1, index].set_ylabel("Standardized value")
        axes[1, index].legend(fontsize=8)
        axes[1, index].grid(True, linestyle=":", alpha=0.5)

        rows.append(
            {
                "column": column,
                "minmax_forward_max_diff": float(np.max(np.abs(manual_minmax_values - sklearn_minmax_values))),
                "minmax_inverse_max_diff": float(
                    max(
                        np.max(np.abs(restored_manual_minmax - values)),
                        np.max(np.abs(restored_sklearn_minmax - values)),
                    )
                ),
                "zscore_forward_max_diff": float(np.max(np.abs(manual_zscore_values - sklearn_zscore_values))),
                "zscore_inverse_max_diff": float(
                    max(
                        np.max(np.abs(restored_manual_zscore - values)),
                        np.max(np.abs(restored_sklearn_zscore - values)),
                    )
                ),
            }
        )
    fig.tight_layout()
    save_figure(fig, results_dir, "34_scaling_comparison")
    plt.close(fig)
    result = pd.DataFrame(rows)
    print("\nScaling comparison (manual vs scikit-learn):")
    print(result.to_string(index=False))
    return result
