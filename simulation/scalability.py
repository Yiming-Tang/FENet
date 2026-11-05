"""Supplementary scalability experiment adapted from the supplied source.

The experiment follows the single-change-point Fourier/IID construction used in
the original simulation notebook, while generating samples in vectorized batches.  It keeps
the submitted practical FENet head (256--64), caches the deterministic Gram
embedding once, and compares N_train=500 and 1,000 on nested training sets.
"""

from __future__ import annotations

import gc
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .fenet import FENet, embedding_from_gram


T_VALUES = (50, 100, 200, 500)
TRAIN_SIZES = (500, 1000)
EXPERIMENT_SEEDS = (101, 202, 303)
N_TEST = 1000
N_EPOCHS = 20
BATCH_SIZE = 64
EMBEDDING_BATCH_SIZE = 16
COEFFICIENT_NOISE_SD = 0.5
TEST_CHANGE_STRENGTH = 0.5


def _basis_functions(x_values: np.ndarray, dimension: int) -> np.ndarray:
    """Fallback copy of the notebook's Fourier-basis constructor."""
    return np.asarray(
        [
            [
                math.sqrt(2.0) * math.sin(order * math.pi * x)
                if order % 2 == 1
                else math.sqrt(2.0) * math.cos(order * math.pi * x)
                for order in range(1, dimension + 1)
            ]
            for x in x_values
        ],
        dtype=np.float32,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


PracticalFENetMLP = FENet


def practical_parameter_count(window_length: int) -> int:
    return 256 * window_length * (window_length - 1) + 16_769


def theoretical_dense_first_layer_count(window_length: int) -> int:
    return window_length * (window_length - 1) ** 2


def estimated_fp32_adam_memory_gib(
    window_length: int,
    batch_size: int = BATCH_SIZE,
) -> float:
    """Parameters, gradients, two Adam moments, and one input batch."""
    input_dimension = window_length * (window_length - 1)
    byte_count = 16 * practical_parameter_count(window_length)
    byte_count += 4 * batch_size * input_dimension
    return byte_count / (1024**3)


def _make_balanced_labels(sample_count: int) -> np.ndarray:
    """Alternating labels keep every even prefix, including 500, balanced."""
    if sample_count % 2:
        raise ValueError("The requested sample count must be even.")
    return (np.arange(sample_count) % 2).astype(np.float32)


def _generate_single_cp_batch(
    window_length: int,
    labels: np.ndarray,
    basis_matrix: np.ndarray,
    rng: np.random.Generator,
    training_distribution: bool,
    coefficient_noise_sd: float = COEFFICIENT_NOISE_SD,
    test_change_strength: float = TEST_CHANGE_STRENGTH,
) -> np.ndarray:
    """Vectorized Scenario-3/IID single-CP DGP from Experiment 1.

    Training positives draw c from U[0.1,1], xi from {1,...,D}, and the
    relative location from U[0.1,0.9]. Test positives use c=0.5,
    xi=floor(D/2), and location 0.5. All coefficient innovations have SD 0.5.
    """
    batch_size = len(labels)
    basis_dimension = basis_matrix.shape[1]
    grid_size = basis_matrix.shape[0]

    coefficients = rng.normal(
        loc=0.0,
        scale=coefficient_noise_sd,
        size=(batch_size, window_length, basis_dimension),
    ).astype(np.float32)
    observations = coefficients @ basis_matrix.T

    for sample_index in np.flatnonzero(labels == 1):
        permutation = rng.permutation(basis_dimension)
        if training_distribution:
            change_strength = float(rng.uniform(0.1, 1.0))
            affected_dimension = int(rng.integers(1, basis_dimension + 1))
            relative_location = float(rng.uniform(0.1, 0.9))
        else:
            change_strength = test_change_strength
            affected_dimension = basis_dimension // 2
            relative_location = 0.5
        selected = permutation[:affected_dimension]
        delta = basis_matrix[:, selected].sum(axis=1)
        delta *= math.sqrt(change_strength / affected_dimension)
        change_index = int(math.floor(relative_location * window_length))
        observations[sample_index, change_index + 1 :, :] += delta.reshape(1, grid_size)

    return observations.astype(np.float32, copy=False)


def _embedding_from_gram(gram: torch.Tensor) -> torch.Tensor:
    """Use the shared paper-convention embedding."""
    return embedding_from_gram(gram, convention="paper")


def _make_cached_features(
    window_length: int,
    sample_count: int,
    seed: int,
    basis_matrix: np.ndarray,
    embedding_device: torch.device,
    embedding_batch_size: int,
    training_distribution: bool,
    coefficient_noise_sd: float = COEFFICIENT_NOISE_SD,
    test_change_strength: float = TEST_CHANGE_STRENGTH,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Generate samples and cache embeddings; time only the embedding step."""
    labels = _make_balanced_labels(sample_count)
    rng = np.random.default_rng(seed)
    feature_chunks: list[torch.Tensor] = []
    embedding_seconds = 0.0

    for start in range(0, sample_count, embedding_batch_size):
        stop = min(start + embedding_batch_size, sample_count)
        label_chunk = labels[start:stop]
        observations = _generate_single_cp_batch(
            window_length,
            label_chunk,
            basis_matrix,
            rng,
            training_distribution,
            coefficient_noise_sd,
            test_change_strength,
        )
        observations_tensor = torch.from_numpy(observations).to(embedding_device)

        _synchronize(embedding_device)
        tic = time.perf_counter()
        gram = observations_tensor @ observations_tensor.transpose(1, 2)
        embedded = _embedding_from_gram(gram)
        _synchronize(embedding_device)
        embedding_seconds += time.perf_counter() - tic

        feature_chunks.append(embedded.detach().cpu())
        del observations_tensor, gram, embedded

    features = torch.cat(feature_chunks, dim=0).contiguous()
    label_tensor = torch.from_numpy(labels)
    return features, label_tensor, embedding_seconds / sample_count


def _evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for feature_batch, _ in loader:
            logits = model(feature_batch.to(device))
            probabilities.append(torch.sigmoid(logits).cpu())
    probability_array = torch.cat(probabilities).numpy()
    label_array = labels.numpy()
    predictions = (probability_array >= 0.5).astype(np.float32)
    return (
        float(accuracy_score(label_array, predictions)),
        float(roc_auc_score(label_array, probability_array)),
    )


@dataclass
class FitResult:
    accuracy: float
    auc: float
    mean_epoch_seconds: float
    peak_device_memory_gb: float


def _fit_one_model(
    window_length: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> FitResult:
    _set_seed(seed)
    model = PracticalFENetMLP(window_length).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    epoch_seconds: list[float] = []

    for _ in range(epochs):
        model.train()
        _synchronize(device)
        tic = time.perf_counter()
        for feature_batch, label_batch in train_loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
        _synchronize(device)
        epoch_seconds.append(time.perf_counter() - tic)

    accuracy, auc = _evaluate(model, test_features, test_labels, device, batch_size)

    peak_memory_gb = float("nan")
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

    measured_epochs = epoch_seconds[1:] if len(epoch_seconds) > 1 else epoch_seconds
    mean_epoch_seconds = float(np.mean(measured_epochs))
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return FitResult(accuracy, auc, mean_epoch_seconds, peak_memory_gb)


def format_markdown_tables(
    computation_summary: pd.DataFrame,
    sample_summary: pd.DataFrame,
) -> tuple[str, str]:
    """Format the two compact Markdown tables for reporting."""

    def mean_sd(mean: float, sd: float, digits: int) -> str:
        if pd.isna(sd):
            return f"{mean:.{digits}f} (NA)"
        return f"{mean:.{digits}f} ({sd:.{digits}f})"

    train_sizes = sorted(int(value) for value in sample_summary["train_size"].unique())
    accuracy_lines = [
        "| $T$ | "
        + " | ".join(f"$N\\_{{\\rm train}}={size}$" for size in train_sizes)
        + " |",
        "|---:|" + "---:|" * len(train_sizes),
    ]
    for window_length in sorted(int(value) for value in sample_summary["T"].unique()):
        row = sample_summary[sample_summary["T"] == window_length].set_index("train_size")
        values = [
            mean_sd(
                float(row.loc[size, "Accuracy_mean"]),
                float(row.loc[size, "Accuracy_std"]),
                4,
            )
            for size in train_sizes
        ]
        accuracy_lines.append(
            f"| {window_length} | " + " | ".join(values) + " |"
        )

    timing_lines = [
        "| $T$ | Input dimension | Practical parameters | Embedding, ms/sample | One epoch, s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in computation_summary.sort_values("T").iterrows():
        timing_lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["T"])),
                    f'{int(row["input_dimension"]):,}',
                    f'{int(row["practical_parameters"]):,}',
                    mean_sd(
                        float(row["embedding_ms_mean"]),
                        float(row["embedding_ms_std"]),
                        3,
                    ),
                    mean_sd(
                        float(row["epoch_seconds_mean"]),
                        float(row["epoch_seconds_std"]),
                        3,
                    ),
                ]
            )
            + " |"
        )

    return "\n".join(accuracy_lines), "\n".join(timing_lines)


def run_scalability_experiment(
    notebook_basis_function=None,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the benchmark and return raw, computational, and sample summaries."""
    device = _select_device()
    basis_constructor = notebook_basis_function or _basis_functions
    grid_size = 50
    basis_dimension = 21
    basis_matrix = np.asarray(
        basis_constructor(np.linspace(0.0, 1.0, grid_size), basis_dimension),
        dtype=np.float32,
    )

    t_values = (50, 100) if quick else T_VALUES
    train_sizes = (500,) if quick else TRAIN_SIZES
    seeds = (101,) if quick else EXPERIMENT_SEEDS
    epochs = 2 if quick else N_EPOCHS
    n_test = 200 if quick else N_TEST
    embedding_batch_size = min(EMBEDDING_BATCH_SIZE, 8) if quick else EMBEDDING_BATCH_SIZE

    print(
        f"Device={device}; T={t_values}; N_train={train_sizes}; "
        f"seeds={seeds}; epochs={epochs}; noise_sd={COEFFICIENT_NOISE_SD}; "
        f"test_c={TEST_CHANGE_STRENGTH}"
    )
    rows: list[dict[str, float | int | str]] = []

    for window_length in t_values:
        for seed in seeds:
            print(f"Preparing cached embeddings: T={window_length}, seed={seed}")
            max_train_size = max(train_sizes)
            train_features, train_labels, embedding_time = _make_cached_features(
                window_length,
                max_train_size,
                seed,
                basis_matrix,
                device,
                embedding_batch_size,
                True,
            )
            test_features, test_labels, _ = _make_cached_features(
                window_length,
                n_test,
                seed + 20_000,
                basis_matrix,
                device,
                embedding_batch_size,
                False,
            )

            for train_size in train_sizes:
                print(f"Training: T={window_length}, N={train_size}, seed={seed}")
                fit = _fit_one_model(
                    window_length,
                    train_features[:train_size],
                    train_labels[:train_size],
                    test_features,
                    test_labels,
                    seed,
                    device,
                    epochs,
                    BATCH_SIZE,
                )
                row = {
                    "device": str(device),
                    "T": window_length,
                    "input_dimension": window_length * (window_length - 1),
                    "practical_parameters": practical_parameter_count(window_length),
                    "theoretical_dense_first_layer": theoretical_dense_first_layer_count(window_length),
                    "estimated_fp32_adam_gib": estimated_fp32_adam_memory_gib(window_length),
                    "train_size": train_size,
                    "seed": seed,
                    "embedding_ms_per_sample": 1000.0 * embedding_time,
                    "epoch_seconds": fit.mean_epoch_seconds,
                    "peak_device_memory_gb": fit.peak_device_memory_gb,
                    "Accuracy": fit.accuracy,
                    "AUC": fit.auc,
                }
                rows.append(row)
                print(
                    f"  Accuracy={fit.accuracy:.4f}, AUC={fit.auc:.4f}, "
                    f"epoch={fit.mean_epoch_seconds:.3f}s"
                )

            del train_features, train_labels
            del test_features, test_labels
            gc.collect()

    raw_results = pd.DataFrame(rows)
    computation_source = raw_results[raw_results["train_size"] == max(train_sizes)]
    computation_summary = (
        computation_source.groupby(
            [
                "T",
                "input_dimension",
                "practical_parameters",
                "estimated_fp32_adam_gib",
            ],
            as_index=False,
        )
        .agg(
            embedding_ms_mean=("embedding_ms_per_sample", "mean"),
            embedding_ms_std=("embedding_ms_per_sample", "std"),
            epoch_seconds_mean=("epoch_seconds", "mean"),
            epoch_seconds_std=("epoch_seconds", "std"),
            peak_memory_gb_mean=("peak_device_memory_gb", "mean"),
            peak_memory_gb_std=("peak_device_memory_gb", "std"),
        )
    )
    sample_summary = (
        raw_results.groupby(["T", "train_size"], as_index=False)
        .agg(
            Accuracy_mean=("Accuracy", "mean"),
            Accuracy_std=("Accuracy", "std"),
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
        )
    )

    print("\nComputational scaling summary")
    print(computation_summary.round(4).to_string(index=False))
    print("\nSample-efficiency summary")
    print(sample_summary.round(4).to_string(index=False))
    accuracy_markdown, timing_markdown = format_markdown_tables(
        computation_summary,
        sample_summary,
    )
    print("\nAccuracy table for reporting\n")
    print(accuracy_markdown)
    print("\nComputational-time table for reporting\n")
    print(timing_markdown)
    return raw_results, computation_summary, sample_summary


def main() -> None:
    """Run the supplied benchmark settings and save CSV summaries."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Reduced experiment; not manuscript results.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/scalability"))
    args = parser.parse_args()
    raw, computation, sample = run_scalability_experiment(quick=args.quick)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw.csv", index=False)
    computation.to_csv(args.output_dir / "computation.csv", index=False)
    sample.to_csv(args.output_dir / "sample_efficiency.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "quick": args.quick, "embedding": "paper",
        "seeds": [101] if args.quick else list(EXPERIMENT_SEEDS),
        "window_lengths": [50, 100] if args.quick else list(T_VALUES),
        "training_sizes": [500] if args.quick else list(TRAIN_SIZES),
        "n_test": 200 if args.quick else N_TEST,
        "epochs": 2 if args.quick else N_EPOCHS,
        "learning_rate": 0.001, "batch_size": BATCH_SIZE,
        "noise_sd": COEFFICIENT_NOISE_SD, "test_change_strength": TEST_CHANGE_STRENGTH,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
