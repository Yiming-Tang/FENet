"""Build the public Sleep-EDF subset as EEG, EOG, or aligned paired tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .preprocessing import SleepEDFConfig, check_fenet_compatibility, prepare_and_save


def verify_modalities(eeg: dict, eog: dict) -> None:
    """Require identical sample order, labels, and transitions across modalities."""
    for split in ("train", "val", "test"):
        for prefix in ("Y", "cp"):
            key = f"{prefix}_{split}"
            if not torch.equal(eeg[key], eog[key]):
                raise AssertionError(f"EEG/EOG are misaligned in {key}.")
        for prefix in ("change_locations", "metadata"):
            key = f"{prefix}_{split}"
            if eeg[key] != eog[key]:
                raise AssertionError(f"EEG/EOG are misaligned in {key}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/sleep-edf"))
    parser.add_argument("--modality", choices=("eeg", "eog", "both"), default="eog")
    parser.add_argument("--nights", choices=("first", "all"), default="all")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    modalities = ("eeg", "eog") if args.modality == "both" else (args.modality,)
    datasets = {}
    for modality in modalities:
        config = SleepEDFConfig(
            data_root=args.data_root.expanduser().resolve(),
            eeg_channel="EEG Fpz-Cz" if modality == "eeg" else "EOG horizontal",
            seed=args.seed,
        )
        prefix = "sleep_edf" + ("_eog" if modality == "eog" else "")
        if args.nights == "all":
            prefix += "_two_nights"
        path, payload = prepare_and_save(
            config,
            output_name=f"{prefix}_logpsd_T{config.window_length}.pt",
            metadata_prefix=prefix,
            nights=args.nights,
        )
        datasets[modality] = payload
        print(f"Saved {modality.upper()}: {path}")
        print(check_fenet_compatibility(payload))
        for split in ("train", "val", "test"):
            labels = payload[f"Y_{split}"]
            print(f"{split}: {len(labels)} windows, {int(labels.sum())} positive")
    if len(datasets) == 2:
        verify_modalities(datasets["eeg"], datasets["eog"])
        print("EEG/EOG labels, transitions, and metadata are exactly aligned.")


if __name__ == "__main__":
    main()
