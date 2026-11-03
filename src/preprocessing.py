from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np


class MinMaxScalerManual:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_: np.ndarray | float | None = None
        self.data_max_: np.ndarray | float | None = None

    def fit(self, values: np.ndarray) -> "MinMaxScalerManual":
        values = np.asarray(values, dtype=float)
        self.data_min_ = np.nanmin(values, axis=0)
        self.data_max_ = np.nanmax(values, axis=0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler is not fitted")
        values = np.asarray(values, dtype=float)
        low, high = self.feature_range
        span = np.asarray(self.data_max_) - np.asarray(self.data_min_)
        span = np.where(span == 0, 1.0, span)
        return low + (values - self.data_min_) * (high - low) / span

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler is not fitted")
        values = np.asarray(values, dtype=float)
        low, high = self.feature_range
        span = np.asarray(self.data_max_) - np.asarray(self.data_min_)
        span = np.where(span == 0, 1.0, span)
        return self.data_min_ + (values - low) * span / (high - low)

    def get_params(self) -> dict[str, object]:
        return {
            "feature_range": self.feature_range,
            "data_min": self.data_min_,
            "data_max": self.data_max_,
        }


class ZScoreScalerManual:
    def __init__(self):
        self.mean_: np.ndarray | float | None = None
        self.std_: np.ndarray | float | None = None

    def fit(self, values: np.ndarray) -> "ZScoreScalerManual":
        values = np.asarray(values, dtype=float)
        self.mean_ = np.nanmean(values, axis=0)
        self.std_ = np.nanstd(values, axis=0, ddof=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        values = np.asarray(values, dtype=float)
        return (values - self.mean_) / self.std_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted")
        values = np.asarray(values, dtype=float)
        return values * self.std_ + self.mean_

    def get_params(self) -> dict[str, object]:
        return {"mean": self.mean_, "std": self.std_}


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    return MinMaxScalerManual((0.0, 1.0)).fit(values).transform(values)


def minmax_normalize_bounds(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return MinMaxScalerManual((low, high)).fit(values).transform(values)


def zscore_standardize(values: np.ndarray) -> np.ndarray:
    return ZScoreScalerManual().fit(values).transform(values)


def split_dataset(values: np.ndarray, ratios: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values)
    shares = np.asarray(list(ratios), dtype=float)
    if shares.size != 3 or np.any(shares <= 0):
        raise ValueError("Exactly three positive ratios are required")
    if shares.sum() > 1.5:
        shares = shares / 100.0
    shares = shares / shares.sum()
    total = len(array)
    train_size = int(round(total * shares[0]))
    val_size = int(round(total * shares[1]))
    if train_size + val_size > total:
        val_size = max(0, total - train_size)
    test_size = total - train_size - val_size
    train = array[:train_size]
    val = array[train_size:train_size + val_size]
    test = array[train_size + val_size:train_size + val_size + test_size]
    return train, val, test


def set_seed(seed: int = 42) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
