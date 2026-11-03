from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def unique_path(directory: str | Path, stem: str, suffix: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{stem}_{timestamp()}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{timestamp()}_{index}{suffix}"
        index += 1
    return candidate


def _autoname_columns(columns: list[Any]) -> list[str]:
    names: list[str] = []
    for index, column in enumerate(columns):
        text = str(column)
        if text.lower().startswith("unnamed") or text in {"nan", "None"}:
            names.append(f"col_{index + 1}")
        elif re.fullmatch(r"\d+(\.0)?", text):
            names.append(f"col_{index + 1}")
        else:
            names.append(text)
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".txt":
        frame = pd.read_csv(path, sep=None, engine="python")
    elif suffix == ".mat":
        content = loadmat(path)
        arrays = [value for key, value in content.items() if not key.startswith("__") and isinstance(value, np.ndarray)]
        matrix = next((value for value in arrays if value.ndim == 2), None)
        if matrix is None:
            raise ValueError(f"No 2D numeric array found in {path}")
        frame = pd.DataFrame(matrix)
        if "columns" in content:
            column_names = []
            for item in np.asarray(content["columns"]).ravel():
                if isinstance(item, np.ndarray):
                    item = item.ravel()[0] if item.size else ""
                column_names.append(str(item))
            if len(column_names) == frame.shape[1]:
                frame.columns = column_names
    else:
        raise ValueError(f"Unsupported format: {suffix}")
    frame.columns = _autoname_columns(list(frame.columns))
    return frame


def save_table(frame: pd.DataFrame, output_dir: str | Path, basename: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    paths: dict[str, Path] = {}

    xlsx_path = output_dir / f"{basename}_{stamp}.xlsx"
    frame.to_excel(xlsx_path, index=False)
    paths["xlsx"] = xlsx_path

    csv_path = output_dir / f"{basename}_{stamp}.csv"
    frame.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    txt_path = output_dir / f"{basename}_{stamp}.txt"
    frame.to_csv(txt_path, sep="\t", index=False)
    paths["txt"] = txt_path

    mat_path = output_dir / f"{basename}_{stamp}.mat"
    try:
        mat_data = frame.to_numpy(dtype=float)
    except (TypeError, ValueError):
        mat_data = frame.astype(str).to_numpy(dtype=object)
    savemat(
        mat_path,
        {
            "data": mat_data,
            "columns": np.array([str(column) for column in frame.columns], dtype=object),
        },
        do_compression=True,
    )
    paths["mat"] = mat_path
    return paths


def save_figure(fig, results_dir: str | Path, basename: str, dpi: int = 150) -> Path:
    path = unique_path(results_dir, basename, ".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _variant_starts(raw: pd.DataFrame) -> list[tuple[int, int]]:
    pattern = re.compile(r"(?:вариант|variant)\s*(\d+)", re.IGNORECASE)
    found: list[tuple[int, int]] = []
    for column_index, value in enumerate(raw.iloc[0].tolist()):
        match = pattern.search(str(value))
        if match:
            found.append((int(match.group(1)), column_index))
    return found


def load_variant_dataset(path: str | Path, variant: int = 1) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    starts = _variant_starts(raw)
    if starts:
        starts_sorted = sorted(starts, key=lambda item: item[1])
        start_lookup = {variant_number: column for variant_number, column in starts_sorted}
        if variant not in start_lookup:
            raise ValueError(f"Variant {variant} was not found; available: {sorted(start_lookup)}")
        start = start_lookup[variant]
        next_positions = [column for number, column in starts_sorted if column > start]
        end = next_positions[0] if next_positions else raw.shape[1]
        width = max(1, end - start)
    else:
        width = 2
        start = (variant - 1) * width

    headers = raw.iloc[1, start:start + width].tolist()
    data = raw.iloc[2:, start:start + width].copy()
    names = _autoname_columns(headers)
    data.columns = names
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(how="all").reset_index(drop=True)
    return data


def validate_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "missing": frame.isna().sum().to_dict(),
        "object_columns": [column for column in frame.columns if frame[column].dtype == object],
        "duplicate_columns": [column for column in frame.columns if list(frame.columns).count(column) > 1],
        "non_finite": {},
    }
    numeric = frame.select_dtypes(include=[np.number])
    for column in numeric.columns:
        values = numeric[column].to_numpy(dtype=float)
        count = int(np.sum(~np.isfinite(values)))
        if count:
            report["non_finite"][column] = count

    has_missing = any(count > 0 for count in report["missing"].values())
    has_object = bool(report["object_columns"])
    has_duplicates = bool(report["duplicate_columns"])
    has_non_finite = bool(report["non_finite"])
    report["has_issues"] = has_missing or has_object or has_duplicates or has_non_finite

    if has_missing:
        print(f"WARNING: empty cells detected: {report['missing']}")
    if has_object:
        print(f"WARNING: object dtype columns detected: {report['object_columns']}")
    if has_duplicates:
        print(f"WARNING: duplicate column names detected: {report['duplicate_columns']}")
    if has_non_finite:
        print(f"WARNING: non-finite values detected: {report['non_finite']}")
    if not report["has_issues"]:
        print("Validation passed: no empty cells, type mismatches or non-finite values.")
    return report


def set_pandas_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("float64")
    return result


def handle_missing_values(
    frame: pd.DataFrame,
    strategy: str | None = None,
    fill_value: float | None = None,
    interactive: bool = False,
) -> pd.DataFrame:
    if not frame.isna().to_numpy().any():
        return frame.copy()

    options = {
        "1": "drop",
        "2": "cast",
        "3": "mean",
        "4": "ffill",
        "5": "bfill",
        "6": "zero",
        "7": "value",
    }
    if interactive:
        print("\nMissing values detected. Choose an action:")
        print("1 - drop rows, 2 - cast types, 3 - fill with mean, 4 - forward fill,")
        print("5 - backward fill, 6 - fill with zeros, 7 - fill with a custom value")
        choice = input("Action [3]: ").strip() or "3"
        strategy = options.get(choice, "mean")
        if strategy == "value" and fill_value is None:
            raw_value = input("Custom value [0]: ").strip()
            fill_value = float(raw_value) if raw_value else 0.0
    strategy = strategy or "mean"
    result = frame.copy()

    if strategy == "drop":
        return result.dropna(how="any").reset_index(drop=True)
    if strategy == "cast":
        for column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        return result
    if strategy == "mean":
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        result[numeric_columns] = result[numeric_columns].fillna(result[numeric_columns].mean())
        return result
    if strategy == "ffill":
        return result.ffill().bfill()
    if strategy == "bfill":
        return result.bfill().ffill()
    if strategy == "zero":
        return result.fillna(0.0)
    if strategy == "value":
        return result.fillna(0.0 if fill_value is None else fill_value)
    raise ValueError(f"Unknown missing value strategy: {strategy}")
