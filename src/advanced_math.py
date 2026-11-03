from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline, interp1d

from .io_operations import save_figure


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.select_dtypes(include=[np.number])


def spectrograms(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    fig, axes = plt.subplots(len(numeric.columns), 1, figsize=(11, 3.4 * len(numeric.columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], numeric.columns):
        values = numeric[column].dropna().to_numpy()
        nperseg = int(min(32, max(8, len(values) // 4)))
        frequencies, times, spectrum = signal.spectrogram(values, fs=1.0, nperseg=nperseg)
        image = ax.pcolormesh(times, frequencies, 10 * np.log10(spectrum + 1e-12), shading="auto", cmap="magma")
        ax.set_title(f"Spectrogram: {column}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        fig.colorbar(image, ax=ax, label="Power (dB)")
    fig.tight_layout()
    path = save_figure(fig, results_dir, "28_spectrograms")
    plt.close(fig)
    return path


def periodograms(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    fig, axes = plt.subplots(len(numeric.columns), 1, figsize=(11, 3.4 * len(numeric.columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], numeric.columns):
        values = numeric[column].dropna().to_numpy()
        nperseg = int(min(32, max(8, len(values) // 4)))
        frequencies, power = signal.periodogram(values, fs=1.0, nfft=nperseg)
        ax.semilogy(frequencies, power + 1e-12)
        ax.set_title(f"Periodogram: {column}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "29_periodograms")
    plt.close(fig)
    return path


def fft_spectra(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    fig, axes = plt.subplots(len(numeric.columns), 1, figsize=(11, 3.4 * len(numeric.columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], numeric.columns):
        values = numeric[column].dropna().to_numpy()
        spectrum = np.fft.rfft(values)
        frequencies = np.fft.rfftfreq(len(values), d=1.0)
        amplitude = 2.0 * np.abs(spectrum) / len(values)
        ax.plot(frequencies, amplitude, linewidth=1.3)
        ax.set_title(f"Amplitude-frequency response: {column}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "30_fft_spectra")
    plt.close(fig)
    return path


def cubic_interpolation(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    fig, axes = plt.subplots(len(numeric.columns), 1, figsize=(11, 3.4 * len(numeric.columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], numeric.columns):
        values = numeric[column].dropna().to_numpy()
        indices = np.arange(len(values))
        dense = np.linspace(indices.min(), indices.max(), max(200, len(values) * 4))
        interpolator = interp1d(indices, values, kind="cubic", fill_value="extrapolate")
        ax.plot(indices, values, "o", markersize=3, label="Source")
        ax.plot(dense, interpolator(dense), linewidth=1.5, label="Cubic interpolation")
        ax.set_title(f"Cubic interpolation: {column}")
        ax.set_xlabel("Index")
        ax.set_ylabel(column)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "31_cubic_interpolation")
    plt.close(fig)
    return path


def spline_interpolation(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    fig, axes = plt.subplots(len(numeric.columns), 1, figsize=(11, 3.4 * len(numeric.columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], numeric.columns):
        values = numeric[column].dropna().to_numpy()
        indices = np.arange(len(values))
        dense = np.linspace(indices.min(), indices.max(), max(200, len(values) * 4))
        spline = CubicSpline(indices, values)
        ax.plot(indices, values, "o", markersize=3, label="Source")
        ax.plot(dense, spline(dense), linewidth=1.5, label="Cubic spline")
        ax.set_title(f"Spline interpolation: {column}")
        ax.set_xlabel("Index")
        ax.set_ylabel(column)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "32_spline_interpolation")
    plt.close(fig)
    return path


def binary_masks(frame: pd.DataFrame, results_dir: str | Path):
    numeric = _numeric(frame)
    columns = list(numeric.columns)
    criteria = ["positive", "negative", "zero", "interval_-1_1"]
    results: dict[str, dict[str, np.ndarray]] = {}
    fig, axes = plt.subplots(len(criteria), len(columns), figsize=(7 * len(columns), 2.2 * len(criteria)), squeeze=False)
    for column_index, column in enumerate(columns):
        values = numeric[column].dropna().to_numpy()
        masks = {
            "positive": values > 0,
            "negative": values < 0,
            "zero": values == 0,
            "interval_-1_1": (values >= -1) & (values <= 1),
        }
        results[column] = masks
        for row_index, name in enumerate(criteria):
            ax = axes[row_index, column_index]
            ax.imshow(masks[name][np.newaxis, :], aspect="auto", cmap="gray_r", interpolation="nearest")
            ax.set_title(f"{column}: {name}")
            ax.set_yticks([])
    fig.tight_layout()
    path = save_figure(fig, results_dir, "33_binary_masks")
    plt.close(fig)
    return results, path


def sliding_window(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values)
    if width < 1 or width > len(values):
        raise ValueError("Window width must be between 1 and the array length")
    return np.lib.stride_tricks.sliding_window_view(values, width)


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    windows = sliding_window(np.asarray(values, dtype=float), width)
    return windows.mean(axis=-1)


def plot_moving_average(frame: pd.DataFrame, width: int = 7, results_dir: str | Path | None = None):
    numeric = _numeric(frame)
    results = {}
    fig, axes = plt.subplots(len(numeric.columns), 1, figsize=(11, 3.4 * len(numeric.columns)), squeeze=False)
    for ax, column in zip(axes[:, 0], numeric.columns):
        values = numeric[column].dropna().to_numpy()
        average = moving_average(values, width)
        results[column] = average
        ax.plot(np.arange(len(values)), values, alpha=0.45, label="Source")
        ax.plot(np.arange(width - 1, len(values)), average, linewidth=1.8, label=f"Moving average ({width})")
        ax.set_title(f"Moving average: {column}")
        ax.set_xlabel("Index")
        ax.set_ylabel(column)
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = save_figure(fig, results_dir, "37_moving_average") if results_dir is not None else None
    plt.close(fig)
    return results, path
