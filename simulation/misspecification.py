"""Supplementary distribution-misspecification stress tests from the supplied source."""

from __future__ import annotations

import gc
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .scalability import (
    BATCH_SIZE,
    EMBEDDING_BATCH_SIZE,
    N_EPOCHS,
    PracticalFENetMLP,
    _basis_functions,
    _embedding_from_gram,
    _make_cached_features,
    _select_device,
    _set_seed,
    _synchronize,
)


MISSPEC_T = 100
MISSPEC_N_TRAIN = 1000
MISSPEC_N_TEST = 1000
MISSPEC_SEEDS = (101, 202, 303)
MISSPEC_THRESHOLD = 0.5
MISSPEC_RAMP_FRACTION = 0.2
MISSPEC_TRAIN_NOISE_SD = 0.20
MISSPEC_IN_FAMILY_NOISE_SD = 0.20
MISSPEC_STRESS_NOISE_SDS = (0.50,)
MISSPEC_T_DF = 3.0


def _balanced_labels(sample_count: int) -> np.ndarray:
    if sample_count % 2:
        raise ValueError("Balanced tests require an even sample count.")
    return (np.arange(sample_count) % 2).astype(np.float32)


def _fixed_test_delta(
    basis_matrix: np.ndarray,
    rng: np.random.Generator,
    change_strength: float,
) -> np.ndarray:
    basis_dimension = basis_matrix.shape[1]
    affected_dimension = basis_dimension // 2
    selected = rng.permutation(basis_dimension)[:affected_dimension]
    delta = basis_matrix[:, selected].sum(axis=1) / math.sqrt(affected_dimension)
    return delta * math.sqrt(change_strength)


def _make_observations(
    condition: str,
    window_length: int,
    labels: np.ndarray,
    basis_matrix: np.ndarray,
    rng: np.random.Generator,
    change_strength: float,
    coefficient_noise_sd: float,
) -> np.ndarray:
    """Generate one batch under a fixed out-of-family test condition."""
    batch_size = len(labels)
    basis_dimension = basis_matrix.shape[1]

    if condition == "heavy_tailed":
        # Var(t_3)=3, so this scaling matches the requested coefficient SD.
        coefficient_noise = rng.standard_t(
            df=MISSPEC_T_DF,
            size=(batch_size, window_length, basis_dimension),
        ).astype(np.float32)
        coefficient_noise *= np.float32(coefficient_noise_sd / math.sqrt(3.0))
    else:
        coefficient_noise = rng.normal(
            loc=0.0,
            scale=coefficient_noise_sd,
            size=(batch_size, window_length, basis_dimension),
        ).astype(np.float32)

    observations = coefficient_noise @ basis_matrix.T

    for sample_index in range(batch_size):
        if labels[sample_index] == 0:
            continue

        delta = _fixed_test_delta(basis_matrix, rng, change_strength)
        if condition == "smooth_transition":
            transition_width = max(2, int(round(MISSPEC_RAMP_FRACTION * window_length)))
            transition_start = int(round(0.5 * window_length - transition_width / 2))
            time_index = np.arange(window_length)
            ramp_weight = np.clip(
                (time_index - transition_start) / transition_width,
                0.0,
                1.0,
            ).astype(np.float32)
            observations[sample_index] += ramp_weight[:, None] * delta[None, :]
        elif condition == "heavy_tailed":
            change_index = int(math.floor(0.5 * window_length))
            observations[sample_index, change_index + 1 :] += delta
        else:
            raise ValueError(f"Unknown condition: {condition}")

    return observations.astype(np.float32, copy=False)


def _make_condition_features(
    condition: str,
    sample_count: int,
    seed: int,
    basis_matrix: np.ndarray,
    device: torch.device,
    change_strength: float,
    coefficient_noise_sd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = _balanced_labels(sample_count)
    rng = np.random.default_rng(seed)
    feature_chunks: list[torch.Tensor] = []

    for start in range(0, sample_count, EMBEDDING_BATCH_SIZE):
        stop = min(start + EMBEDDING_BATCH_SIZE, sample_count)
        label_chunk = labels[start:stop]
        observations = _make_observations(
            condition,
            MISSPEC_T,
            label_chunk,
            basis_matrix,
            rng,
            change_strength,
            coefficient_noise_sd,
        )
        observation_tensor = torch.from_numpy(observations).to(device)
        gram = observation_tensor @ observation_tensor.transpose(1, 2)
        embedded = _embedding_from_gram(gram)
        feature_chunks.append(embedded.detach().cpu())
        del observation_tensor, gram, embedded

    return torch.cat(feature_chunks, dim=0).contiguous(), torch.from_numpy(labels)


def _train_frozen_model(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, float]:
    _set_seed(seed)
    model = PracticalFENetMLP(MISSPEC_T).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    epoch_times: list[float] = []

    for _ in range(N_EPOCHS):
        model.train()
        _synchronize(device)
        tic = time.perf_counter()
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
        _synchronize(device)
        epoch_times.append(time.perf_counter() - tic)

    measured = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    return model, float(np.mean(measured))


def _evaluate_condition(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    probability_chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for feature_batch, _ in loader:
            probability_chunks.append(
                torch.sigmoid(model(feature_batch.to(device))).cpu()
            )
    probabilities = torch.cat(probability_chunks).numpy()
    label_array = labels.numpy()
    predictions = (probabilities >= MISSPEC_THRESHOLD).astype(np.float32)
    negative_mask = label_array == 0
    positive_mask = label_array == 1
    metrics = {
        "Accuracy": float(accuracy_score(label_array, predictions)),
        "FAR": float(predictions[negative_mask].mean()),
    }
    metrics["TPR"] = (
        float(predictions[positive_mask].mean())
        if positive_mask.any()
        else float("nan")
    )
    metrics["AUC"] = (
        float(roc_auc_score(label_array, probabilities))
        if positive_mask.any() and negative_mask.any()
        else float("nan")
    )
    return metrics


def format_misspecification_markdown(summary: pd.DataFrame) -> str:
    """Format the compact misspecification table for reporting."""

    def mean_sd(row: pd.Series, metric: str) -> str:
        mean_value = float(row[f"{metric}_mean"])
        std_value = float(row[f"{metric}_std"])
        std_text = "NA" if pd.isna(std_value) else f"{std_value:.4f}"
        return f"{mean_value:.4f} ({std_text})"

    rows_by_condition = summary.set_index("condition")
    table_rows = [
        ("Piecewise-constant reference", "Piecewise-constant reference", 0.20, 1.0),
    ]
    for noise_sd in MISSPEC_STRESS_NOISE_SDS:
        for change_strength in (0.5, 0.75):
            strength_label = f"{change_strength:g}"
            noise_label = f"{noise_sd:.2f}"
            table_rows.append(
                (
                    f"smooth_transition_c{strength_label}_noise{noise_label}",
                    "Smooth persistent transition",
                    noise_sd,
                    change_strength,
                )
            )
    for noise_sd in MISSPEC_STRESS_NOISE_SDS:
        for change_strength in (0.5, 0.75):
            strength_label = f"{change_strength:g}"
            noise_label = f"{noise_sd:.2f}"
            table_rows.append(
                (
                    f"heavy_tailed_c{strength_label}_noise{noise_label}",
                    "Student-$t\\_3$ innovations",
                    noise_sd,
                    change_strength,
                )
            )

    lines = [
        "| Test distribution | $c$ | Accuracy | AUC |",
        "|:---|---:|---:|---:|",
    ]
    for condition, display_name, noise_sd, change_strength in table_rows:
        row = rows_by_condition.loc[condition]
        lines.append(
            f"| {display_name} | {change_strength:g} | "
            f"{mean_sd(row, 'Accuracy')} | {mean_sd(row, 'AUC')} |"
        )
    return "\n".join(lines)


def run_misspecification_experiment(
    notebook_basis_function=None,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train in-family once per seed, freeze, and evaluate every OOD condition."""
    device = _select_device()
    basis_constructor = notebook_basis_function or _basis_functions
    basis_matrix = np.asarray(
        basis_constructor(np.linspace(0.0, 1.0, 50), 21),
        dtype=np.float32,
    )
    seeds = (101,) if quick else MISSPEC_SEEDS
    n_train = 200 if quick else MISSPEC_N_TRAIN
    n_test = 200 if quick else MISSPEC_N_TEST
    rows: list[dict[str, float | int | str]] = []

    print(
        f"Misspecification experiment: device={device}, T={MISSPEC_T}, "
        f"N_train={n_train}, N_test={n_test}, seeds={seeds}, "
        f"train_noise_sd={MISSPEC_TRAIN_NOISE_SD}, "
        f"reference_noise_sd={MISSPEC_IN_FAMILY_NOISE_SD}, "
        f"stress_noise_sds={MISSPEC_STRESS_NOISE_SDS}"
    )
    for seed in seeds:
        train_features, train_labels, _ = _make_cached_features(
            MISSPEC_T,
            n_train,
            seed,
            basis_matrix,
            device,
            EMBEDDING_BATCH_SIZE,
            True,
            coefficient_noise_sd=MISSPEC_TRAIN_NOISE_SD,
        )
        model, epoch_seconds = _train_frozen_model(
            train_features,
            train_labels,
            seed,
            device,
        )
        del train_features, train_labels

        in_family_features, in_family_labels, _ = _make_cached_features(
            MISSPEC_T,
            n_test,
            seed + 20_000,
            basis_matrix,
            device,
            EMBEDDING_BATCH_SIZE,
            False,
            coefficient_noise_sd=MISSPEC_IN_FAMILY_NOISE_SD,
            test_change_strength=1.0,
        )
        in_family_metrics = _evaluate_condition(
            model,
            in_family_features,
            in_family_labels,
            device,
        )
        rows.append(
            {
                "seed": seed,
                "condition": "Piecewise-constant reference",
                "epoch_seconds": epoch_seconds,
                **in_family_metrics,
            }
        )
        del in_family_features, in_family_labels

        for condition, seed_offset, change_strength in (
            ("smooth_transition", 30_000, 0.5),
            ("smooth_transition", 30_000, 0.75),
            ("heavy_tailed", 40_000, 0.5),
            ("heavy_tailed", 40_000, 0.75),
        ):
            for coefficient_noise_sd in MISSPEC_STRESS_NOISE_SDS:
                strength_label = f"{change_strength:g}"
                noise_label = f"{coefficient_noise_sd:.2f}"
                result_name = (
                    f"{condition}_c{strength_label}_noise{noise_label}"
                )
                features, labels = _make_condition_features(
                    condition,
                    n_test,
                    seed + seed_offset,
                    basis_matrix,
                    device,
                    change_strength,
                    coefficient_noise_sd,
                )
                metrics = _evaluate_condition(model, features, labels, device)
                rows.append(
                    {
                        "seed": seed,
                        "condition": result_name,
                        "epoch_seconds": epoch_seconds,
                        **metrics,
                    }
                )
                del features, labels

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    raw_results = pd.DataFrame(rows)
    summary = (
        raw_results.groupby("condition", as_index=False)
        .agg(
            Accuracy_mean=("Accuracy", "mean"),
            Accuracy_std=("Accuracy", "std"),
            TPR_mean=("TPR", "mean"),
            TPR_std=("TPR", "std"),
            FAR_mean=("FAR", "mean"),
            FAR_std=("FAR", "std"),
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", "std"),
        )
    )
    print("\nRaw seed-level results")
    print(raw_results.round(4).to_string(index=False))
    print("\nMisspecification summary")
    print(summary.round(4).to_string(index=False))
    print("\nMarkdown table for reporting\n")
    print(format_misspecification_markdown(summary))
    return raw_results, summary


def main() -> None:
    """Run the supplied stress tests and save seed-level results and summaries."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="One seed and 200 samples; training remains 20 epochs.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/misspecification"))
    args = parser.parse_args()
    raw, summary = run_misspecification_experiment(quick=args.quick)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "raw.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "quick": args.quick, "embedding": "paper", "epochs": N_EPOCHS,
        "seeds": [101] if args.quick else list(MISSPEC_SEEDS),
        "n_train": 200 if args.quick else MISSPEC_N_TRAIN,
        "n_test": 200 if args.quick else MISSPEC_N_TEST,
        "learning_rate": 0.001, "batch_size": BATCH_SIZE,
        "train_noise_sd": MISSPEC_TRAIN_NOISE_SD,
        "reference_noise_sd": MISSPEC_IN_FAMILY_NOISE_SD,
        "stress_noise_sds": list(MISSPEC_STRESS_NOISE_SDS),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
