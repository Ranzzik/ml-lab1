from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import advanced_math, data_analysis, image_processing, io_operations, tensors_pca
from src.preprocessing import (
    MinMaxScalerManual,
    ZScoreScalerManual,
    set_seed,
    split_dataset,
)


def section(number: int, title: str) -> None:
    print(f"\n{'=' * 80}\nПункт {number}. {title}\n{'=' * 80}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laboratory work 1: Python ML libraries and data processing")
    parser.add_argument("--data", type=str, default=None, help="Path to the source .xlsx dataset")
    parser.add_argument("--variant", type=int, default=1, help="Variant number")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interactive", action="store_true", help="Enable the missing-value action menu")
    parser.add_argument(
        "--missing-strategy",
        choices=["drop", "cast", "mean", "ffill", "bfill", "zero", "value"],
        default="mean",
        help="Non-interactive missing value strategy",
    )
    parser.add_argument("--fill-value", type=float, default=0.0, help="Value for the custom fill strategy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"
    results_dir = base_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data) if args.data else base_dir / "data" / "Lab1_Dataset.xlsx"
    if not data_path.exists():
        fallback = base_dir / "Lab1_Dataset.xlsx"
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(f"Dataset not found: {data_path}")

    section(13, "Фиксация seed для всех генераторов случайных чисел")
    set_seed(args.seed)
    print(f"Seed fixed: {args.seed}")

    section(1, "Чтение файлов .xlsx, .csv, .txt, .mat")
    source = io_operations.read_table(data_path)
    print(f"Source xlsx: shape={source.shape}, first columns={list(source.columns[:4])}")

    section(6, "Извлечение столбцов варианта и отбрасывание остальных данных")
    frame = io_operations.load_variant_dataset(data_path, args.variant)
    print(f"Variant {args.variant}: shape={frame.shape}, columns={list(frame.columns)}")
    print("Original file was not modified.")

    section(4, "Проверка пустых ячеек, некорректных данных и несовпадения типов")
    validation = io_operations.validate_dataframe(frame)
    print(f"Validation summary: {validation}")

    section(5, "Вывод содержимого таблицы")
    print(frame.to_string())

    section(7, "Указание типов данных Pandas для считанных столбцов")
    frame = io_operations.set_pandas_dtypes(frame)
    print(frame.dtypes.to_string())

    section(8, "Обработка пустых ячеек")
    before_missing = int(frame.isna().sum().sum())
    frame = io_operations.handle_missing_values(
        frame,
        strategy=args.missing_strategy,
        fill_value=args.fill_value,
        interactive=args.interactive,
    )
    after_missing = int(frame.isna().sum().sum())
    print(f"Missing cells before={before_missing}, after={after_missing}, strategy={args.missing_strategy}")

    section(2, "Запись табличных файлов .xlsx, .csv, .txt, .mat с датой и временем")
    saved_paths = io_operations.save_table(frame, output_dir, f"variant_{args.variant}_clean")
    for format_name, path in saved_paths.items():
        restored = io_operations.read_table(path)
        print(f"Saved {format_name}: {path.name}; restored shape={restored.shape}, columns={list(restored.columns)}")

    section(3, "Сохранение изображения .png с датой и временем")
    preview_fig, preview_ax = plt.subplots(figsize=(9, 4.5))
    for column in frame.columns:
        preview_ax.plot(np.arange(len(frame)), frame[column].to_numpy(), label=str(column))
    preview_ax.set_title("Variant data preview")
    preview_ax.set_xlabel("Index")
    preview_ax.set_ylabel("Value")
    preview_ax.legend()
    preview_ax.grid(True, linestyle=":", alpha=0.5)
    preview_fig.tight_layout()
    preview_path = io_operations.save_figure(preview_fig, results_dir, "03_data_preview")
    plt.close(preview_fig)
    print(f"Saved image: {preview_path.name}")

    section(9, "Конвертация DataFrame в массив NumPy")
    data = frame.to_numpy(dtype=float)
    print(f"NumPy array: shape={data.shape}, dtype={data.dtype}")

    section(10, "Ручные функции масштабирования MinMax, MinMax(a,b) и Z-score")
    scaler_minmax = MinMaxScalerManual((0.0, 1.0)).fit(data)
    scaler_bounds = MinMaxScalerManual((-1.0, 1.0)).fit(data)
    scaler_zscore = ZScoreScalerManual().fit(data)
    data_minmax = scaler_minmax.transform(data)
    data_bounds = scaler_bounds.transform(data)
    data_zscore = scaler_zscore.transform(data)
    print(f"MinMax params: min={scaler_minmax.data_min_}, max={scaler_minmax.data_max_}")
    print(f"Bounds MinMax params: min={scaler_bounds.data_min_}, max={scaler_bounds.data_max_}")
    print(f"Z-score params: mean={scaler_zscore.mean_}, std={scaler_zscore.std_}")
    print(f"MinMax range: [{data_minmax.min():.4f}, {data_minmax.max():.4f}]")
    print(f"Bounds range: [{data_bounds.min():.4f}, {data_bounds.max():.4f}]")
    print(f"Z-score range: [{data_zscore.min():.4f}, {data_zscore.max():.4f}]")

    section(11, "Восстановление исходного масштаба из сохранённых параметров")
    restored_minmax = scaler_minmax.inverse_transform(data_minmax)
    restored_bounds = scaler_bounds.inverse_transform(data_bounds)
    restored_zscore = scaler_zscore.inverse_transform(data_zscore)
    print(f"MinMax inverse max error: {np.max(np.abs(restored_minmax - data)):.3e}")
    print(f"Bounds inverse max error: {np.max(np.abs(restored_bounds - data)):.3e}")
    print(f"Z-score inverse max error: {np.max(np.abs(restored_zscore - data)):.3e}")

    section(12, "Разбиение массива на train/val/test")
    train, val, test = split_dataset(data_minmax, (0.70, 0.15, 0.15))
    print(f"train shape={train.shape}, val shape={val.shape}, test shape={test.shape}")

    section(14, "График исходных данных")
    print(data_analysis.plot_raw_data(frame, results_dir).name)

    section(15, "Нормализованная гистограмма плотности распределения")
    print(data_analysis.plot_histograms(frame, results_dir).name)

    section(16, "Отсортированные по возрастанию столбцы")
    data_analysis.sorted_columns(frame)

    section(17, "Эмпирическая функция распределения")
    print(data_analysis.plot_ecdf(frame, results_dir).name)

    section(18, "Выборочное среднее, дисперсия, мода и медиана")
    statistics = data_analysis.compute_statistics(frame)
    print(statistics.to_string(index=False))
    data_analysis.plot_statistics(statistics, results_dir)
    io_operations.save_table(statistics, output_dir, "18_statistics")

    section(19, "Доверительные интервалы для среднего и дисперсии, alpha=0.05")
    intervals = data_analysis.confidence_intervals(frame, alpha=0.05)
    print(intervals.to_string(index=False))
    data_analysis.plot_confidence_intervals(intervals, results_dir)
    io_operations.save_table(intervals, output_dir, "19_confidence_intervals")

    section(20, "Ковариация и корреляция столбцов")
    covariance, correlation, _ = data_analysis.covariance_correlation(frame, results_dir)
    print("Covariance:")
    print(covariance.to_string())
    print("Correlation:")
    print(correlation.to_string())
    io_operations.save_table(covariance.reset_index().rename(columns={"index": "column"}), output_dir, "20_covariance")
    io_operations.save_table(correlation.reset_index().rename(columns={"index": "column"}), output_dir, "20_correlation")

    section(21, "Проверка значимости коэффициента корреляции средствами SciPy")
    significance = data_analysis.correlation_significance(frame)
    io_operations.save_table(significance, output_dir, "21_correlation_significance")

    section(22, "Взаимная корреляция в зависимости от числа сдвиговых элементов")
    _, cross_path = data_analysis.cross_correlation(frame, max_lags=30, results_dir=results_dir)
    print(f"Saved: {cross_path.name}")

    section(23, "Численная производная и градиент")
    derivatives, gradient, _ = data_analysis.numerical_derivative_gradient(frame, results_dir)
    derivative_frame = pd.DataFrame(
        derivatives,
        columns=[f"derivative_{column}" for column in frame.columns],
    )
    io_operations.save_table(derivative_frame, output_dir, "23_derivatives")

    section(24, "Свёртка столбцов")
    convolutions, _ = data_analysis.convolution(frame, results_dir)
    print(f"Convolution results: {list(convolutions)}")

    section(25, "Скалярное и векторное произведения столбцов")
    dot_products, cross_products = data_analysis.dot_and_cross_products(frame)
    io_operations.save_table(dot_products, output_dir, "25_dot_products")
    io_operations.save_table(cross_products, output_dir, "25_cross_products")

    section(26, "L1 и L2 нормы столбцов")
    norms = data_analysis.column_norms(frame)
    io_operations.save_table(norms, output_dir, "26_norms")

    section(27, "Проверка гипотез о равномерном и нормальном распределении")
    tests = data_analysis.distribution_tests(frame)
    io_operations.save_table(tests, output_dir, "27_distribution_tests")

    section(28, "Спектрограммы столбцов")
    print(advanced_math.spectrograms(frame, results_dir).name)

    section(29, "Периодограммы столбцов")
    print(advanced_math.periodograms(frame, results_dir).name)

    section(30, "Амплитудно-частотные характеристики через преобразование Фурье")
    print(advanced_math.fft_spectra(frame, results_dir).name)

    section(31, "Кубическая интерполяция столбцов")
    print(advanced_math.cubic_interpolation(frame, results_dir).name)

    section(32, "Интерполяция сплайнами")
    print(advanced_math.spline_interpolation(frame, results_dir).name)

    section(33, "Бинарные маски столбцов")
    masks, _ = advanced_math.binary_masks(frame, results_dir)
    mask_frame = pd.DataFrame(
        {
            f"{column}_{mask_name}": mask_values.astype(int)
            for column, column_masks in masks.items()
            for mask_name, mask_values in column_masks.items()
        }
    )
    io_operations.save_table(mask_frame, output_dir, "33_binary_masks")

    section(34, "Сравнение ручного масштабирования с MinMaxScaler и StandardScaler")
    scaling_table = data_analysis.scaling_comparison(frame, results_dir)
    io_operations.save_table(scaling_table, output_dir, "34_scaling_comparison")

    section(35, "Сравнение обратного масштабирования с scikit-learn")
    inverse_table = scaling_table[
        [
            "column",
            "minmax_inverse_max_diff",
            "zscore_inverse_max_diff",
        ]
    ]
    print(inverse_table.to_string(index=False))
    io_operations.save_table(inverse_table, output_dir, "35_inverse_scaling")

    section(36, "Функция формирования скользящего окна")
    first_column = frame.columns[0]
    window = advanced_math.sliding_window(frame[first_column].to_numpy(), 7)
    print(f"Sliding window: shape={window.shape}, first window={window[0]}")

    section(37, "Скользящее среднее")
    averages, _ = advanced_math.plot_moving_average(frame, width=7, results_dir=results_dir)
    print(f"Moving averages: { {column: values.shape for column, values in averages.items()} }")

    section(38, "Тензор TensorFlow из исходных данных")
    tensorflow_input = tensors_pca.tensorflow_tensor(frame)

    section(39, "Тензор PyTorch из исходных данных")
    pytorch_input = tensors_pca.pytorch_tensor(frame)

    section(40, "Генерация тензоров A, W, B для операции AXW+B")
    matrix_a, matrix_w, matrix_b = tensors_pca.generate_tensor_factors(frame.shape[0], frame.shape[1], args.seed)
    print(f"A shape={matrix_a.shape}, W shape={matrix_w.shape}, B shape={matrix_b.shape}")

    section(41, "Операция AXW+B средствами TensorFlow")
    tensorflow_result = tensors_pca.tensorflow_axw_b(frame, args.seed)

    section(42, "Операция AXW+B средствами PyTorch")
    pytorch_result = tensors_pca.pytorch_axw_b(frame, args.seed)

    section(43, "Матрицы T, P, Q размерности 3x10")
    matrices = tensors_pca.make_tpq(args.seed)
    for name, matrix in zip(["T", "P", "Q"], matrices):
        print(f"{name}: shape={matrix.shape}, mean={matrix.mean():.4f}")

    section(44, "Выражение V=(sin(T)-exp(P))*Q через Keras.ops")
    keras_result = tensors_pca.keras_expression(matrices)

    section(45, "Выражение V=(sin(T)-exp(P))*Q через PyTorch")
    torch_result = tensors_pca.torch_expression(matrices)
    keras_array = np.asarray(keras_result)
    torch_array = torch_result.detach().cpu().numpy()
    print(f"Keras vs PyTorch max absolute difference: {np.max(np.abs(keras_array - torch_array)):.3e}")
    print("Keras result:")
    print(keras_array)
    print("PyTorch result:")
    print(torch_array)

    section(46, "Разреженная матрица 100000x100000 средствами SciPy")
    sparse_result, sparse_path = tensors_pca.sparse_matrix(output_dir, args.seed)

    section(47, "Обратная матрица случайной матрицы 100x100")
    random_matrix, inverse_matrix, determinant = tensors_pca.random_matrix_inverse(args.seed)
    print(f"Determinant check: {determinant:.6e}; inverse available={inverse_matrix is not None}")

    section(48, "Многомерный массив F из исходных, масштабированных данных и масок")
    tensor_f = tensors_pca.build_tensor_f(frame, masks)

    section(49, "Пайплайн scikit-learn: масштабирование + PCA и логарифмическое правдоподобие")
    experiments, best_projection, likelihood_path, projection_path = tensors_pca.pca_pipeline_experiments(
        tensor_f,
        results_dir,
    )
    print(f"Best projection: {best_projection}")
    print(f"Saved: {likelihood_path.name}, {projection_path.name}")
    io_operations.save_table(experiments, output_dir, "49_pca_experiments")

    section(50, "Обработка цветного изображения средствами SciPy/NumPy")
    image_path = base_dir / "data" / "test_image.png"
    image_paths = image_processing.run_image_pipeline(image_path, output_dir, args.seed)
    for name, path in image_paths.items():
        print(f"{name}: {path.name}")

    print("\n" + "=" * 80)
    print("Все 50 пунктов выполнены. Результаты сохранены в output/ и results/.")
    print(f"Output files: {len(list(output_dir.glob('*')))}")
    print(f"Result files: {len(list(results_dir.glob('*')))}")
    print("=" * 80)


if __name__ == "__main__":
    main()
