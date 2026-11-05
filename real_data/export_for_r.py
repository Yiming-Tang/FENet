"""Export the processed Sleep-EDF test split for the R FF/FPCA methods.

Each sample CSV has shape (T, J): rows are consecutive 30-second epochs and
columns are log-PSD values on the saved frequency grid. In R, Data2fd must
therefore receive ``y = t(sample_matrix)`` so that its columns are the T
functional observations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


DEFAULT_ROOT = Path("data/sleep-edf")
DEFAULT_INPUT = DEFAULT_ROOT / "processed" / "sleep_edf_eog_two_nights_logpsd_T20.pt"
DEFAULT_OUTPUT = DEFAULT_ROOT / "processed" / "r_ff_fpca_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Prepared dataset; defaults to two-night EOG.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for R input CSVs.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    return parser.parse_args()


def write_single_column(path: Path, values: np.ndarray, fmt: str) -> None:
    np.savetxt(path, np.asarray(values).reshape(-1, 1), delimiter=",", fmt=fmt)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    split = args.split
    x = payload[f"X_{split}"].detach().cpu().float().numpy()
    y = payload[f"Y_{split}"].detach().cpu().numpy().astype(np.int64)
    cp = payload[f"cp_{split}"].detach().cpu().numpy().astype(np.int64)
    frequency = payload["frequency_grid_hz"].detach().cpu().float().numpy()
    metadata = payload[f"metadata_{split}"]

    if x.ndim != 3:
        raise ValueError(f"Expected X_{split} to have shape (T, J, N), got {x.shape}.")
    t_size, j_size, n_samples = x.shape
    if y.shape != (n_samples,) or cp.shape != (n_samples,):
        raise ValueError("Labels/change locations do not match the number of samples.")
    if frequency.shape != (j_size,):
        raise ValueError("The frequency grid does not match the functional grid size.")
    if len(metadata) != n_samples:
        raise ValueError("Metadata length does not match the number of samples.")
    if not np.isfinite(x).all():
        raise ValueError("The functional data contain non-finite values.")

    sample_names = []
    for index in range(n_samples):
        sample_name = f"sample_{index + 1:04d}.csv"
        sample_names.append(sample_name)
        np.savetxt(
            output_dir / sample_name,
            x[:, :, index],
            delimiter=",",
            fmt="%.9g",
        )

    write_single_column(output_dir / "Y.csv", y, "%d")
    write_single_column(output_dir / "change_location.csv", cp, "%d")
    write_single_column(output_dir / "frequency_grid_hz.csv", frequency, "%.9g")

    manifest_fields = [
        "sample_file",
        "label",
        "change_location",
        "subject",
        "night",
        "start_second",
        "stage_from",
        "stage_to",
    ]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for index, row in enumerate(metadata):
            writer.writerow(
                {
                    "sample_file": sample_names[index],
                    "label": int(y[index]),
                    "change_location": int(cp[index]),
                    "subject": row["subject"],
                    "night": row["night"],
                    "start_second": row["start_second"],
                    "stage_from": row["stage_from"],
                    "stage_to": row["stage_to"],
                }
            )

    print(f"Exported {n_samples} {split} samples to {output_dir}")
    print(f"Each sample: ({t_size}, {j_size}) = (epochs, frequency bins)")
    print(f"Positive labels: {int(y.sum())}; negative labels: {int((y == 0).sum())}")


if __name__ == "__main__":
    main()
