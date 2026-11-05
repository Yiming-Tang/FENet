"""Prepare Sleep-EDF Expanded as functional time-series windows for FENet.

Each functional observation is the log power-spectral-density (log-PSD) curve
of one 30-second EEG epoch.  Consecutive epochs form a functional time series.
The exported tensors follow the original functional time-series layout:

    X_split.shape == (T, J, N_split)
    Y_split.shape == (N_split,)

where T is the number of epochs per window, J is the frequency-grid size, and
N_split is the number of windows.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal


STAGE_TO_INT = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

STAGE_NAME = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "N4", 5: "REM"}


@dataclass(frozen=True)
class RecordPair:
    subject: str
    night: str
    psg_path: Path
    hypnogram_path: Path


@dataclass
class SleepEDFConfig:
    data_root: Path
    eeg_channel: str = "EEG Fpz-Cz"
    epoch_seconds: int = 30
    window_length: int = 20
    frequency_min_hz: float = 0.5
    frequency_max_hz: float = 30.0
    frequency_bins: int = 120
    welch_segment_seconds: float = 4.0
    wake_buffer_minutes: int = 30
    merge_n3_n4: bool = True
    min_change_margin: int = 3
    window_stride: int = 5
    negative_to_positive_ratio: float | None = None
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    seed: int = 2026
    normalization_clip: float = 8.0

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw" / "sleep-cassette"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def frequency_grid(self) -> np.ndarray:
        return np.linspace(
            self.frequency_min_hz,
            self.frequency_max_hz,
            self.frequency_bins,
            dtype=np.float64,
        )

    def validate(self) -> None:
        fractions = self.train_fraction + self.validation_fraction + self.test_fraction
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("All split fractions must be positive.")
        if not np.isclose(fractions, 1.0):
            raise ValueError(f"Split fractions must sum to one, not {fractions}.")
        if self.window_length < 3:
            raise ValueError("window_length must be at least 3.")
        if not 1 <= self.min_change_margin < self.window_length / 2:
            raise ValueError("min_change_margin must lie inside the window.")
        if self.window_stride < 1:
            raise ValueError("window_stride must be at least one epoch.")
        if (
            self.negative_to_positive_ratio is not None
            and self.negative_to_positive_ratio < 0
        ):
            raise ValueError("negative_to_positive_ratio cannot be negative.")
        if self.epoch_seconds <= 0 or self.frequency_bins < 2:
            raise ValueError("epoch_seconds must be positive and frequency_bins at least two.")
        if self.normalization_clip <= 0:
            raise ValueError("normalization_clip must be positive.")
        if self.frequency_min_hz <= 0 or self.frequency_max_hz <= self.frequency_min_hz:
            raise ValueError("Invalid frequency interval.")


def _load_pyedflib(data_root: Path):
    """Load pyEDFlib from the active Python environment."""
    try:
        import pyedflib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyEDFlib is required for EDF reading. Install the sleep extra: "
            "python -m pip install -e '.[sleep]'."
        ) from exc
    return pyedflib


def discover_record_pairs(raw_dir: Path) -> List[RecordPair]:
    """Pair each downloaded PSG with its matching scored hypnogram."""
    psg_files = sorted(raw_dir.glob("*-PSG.edf"))
    hypnograms = sorted(raw_dir.glob("*-Hypnogram.edf"))
    pairs: List[RecordPair] = []
    for psg_path in psg_files:
        # Sleep Cassette names share their first six characters within a night,
        # while the following technician code varies (EC, EH, EJ, ...).
        prefix = psg_path.name[:6]
        matches = [p for p in hypnograms if p.name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one hypnogram for {psg_path.name}, found {[p.name for p in matches]}."
            )
        pairs.append(
            RecordPair(
                subject=psg_path.name[:5],
                night=prefix,
                psg_path=psg_path,
                hypnogram_path=matches[0],
            )
        )
    if not pairs:
        raise FileNotFoundError(f"No PSG/Hypnogram pairs were found in {raw_dir}.")
    return pairs


def select_record_pairs(pairs: Sequence[RecordPair], nights: str = "all") -> List[RecordPair]:
    """Select all available nights or the first available night per subject."""
    if nights not in {"all", "first"}:
        raise ValueError("nights must be 'all' or 'first'.")
    ordered = sorted(pairs, key=lambda pair: (pair.subject, pair.night))
    if nights == "all":
        return ordered
    selected = {}
    for pair in ordered:
        selected.setdefault(pair.subject, pair)
    return list(selected.values())


def split_subjects(
    subjects: Sequence[str], config: SleepEDFConfig
) -> Dict[str, List[str]]:
    """Create deterministic, mutually exclusive subject-level splits."""
    unique_subjects = np.array(sorted(set(subjects)), dtype=object)
    if len(unique_subjects) < 5:
        raise ValueError("At least five subjects are recommended for train/val/test splitting.")
    rng = np.random.default_rng(config.seed)
    rng.shuffle(unique_subjects)

    n_subjects = len(unique_subjects)
    n_test = max(1, int(round(n_subjects * config.test_fraction)))
    n_validation = max(1, int(round(n_subjects * config.validation_fraction)))
    if n_test + n_validation >= n_subjects:
        raise ValueError("Not enough subjects for the requested split fractions.")

    test = unique_subjects[:n_test].tolist()
    validation = unique_subjects[n_test : n_test + n_validation].tolist()
    train = unique_subjects[n_test + n_validation :].tolist()
    result = {"train": train, "validation": validation, "test": test}

    sets = [set(v) for v in result.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Subject leakage detected between splits.")
    return result


def _read_annotations(hypnogram_path: Path, pyedflib) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    reader = pyedflib.EdfReader(str(hypnogram_path))
    try:
        start_datetime = reader.getStartdatetime()
        onsets, durations, descriptions = reader.readAnnotations()
    finally:
        reader.close()
    descriptions = np.asarray(
        [d.decode("utf-8") if isinstance(d, bytes) else str(d) for d in descriptions],
        dtype=object,
    )
    return start_datetime, np.asarray(onsets), np.asarray(durations), descriptions


def _expand_scored_epochs(
    psg_start_datetime,
    hypnogram_start_datetime,
    onsets: np.ndarray,
    durations: np.ndarray,
    descriptions: np.ndarray,
    config: SleepEDFConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand duration annotations into consecutive 30-second scored epochs."""
    offset_seconds = (hypnogram_start_datetime - psg_start_datetime).total_seconds()
    epoch_onsets: List[float] = []
    stages: List[int] = []

    for onset, duration, description in zip(onsets, durations, descriptions):
        if description not in STAGE_TO_INT:
            # Movement time and unknown stages are deliberately excluded.
            continue
        stage = STAGE_TO_INT[description]
        if config.merge_n3_n4 and stage == 4:
            stage = 3
        epoch_count = int(np.floor(float(duration) / config.epoch_seconds + 1e-8))
        for epoch_index in range(epoch_count):
            epoch_onsets.append(
                offset_seconds + float(onset) + epoch_index * config.epoch_seconds
            )
            stages.append(stage)

    epoch_onsets_array = np.asarray(epoch_onsets, dtype=np.float64)
    stages_array = np.asarray(stages, dtype=np.int8)
    order = np.argsort(epoch_onsets_array)
    epoch_onsets_array = epoch_onsets_array[order]
    stages_array = stages_array[order]

    # Keep a limited amount of wake before the first and after the last sleep
    # epoch. This is the standard Sleep-EDF trimming convention and prevents
    # many hours of wake from dominating the negative class.
    sleep_indices = np.flatnonzero(stages_array != 0)
    if len(sleep_indices) == 0:
        raise ValueError("The hypnogram contains no scored sleep epochs.")
    buffer_seconds = config.wake_buffer_minutes * 60
    lower = epoch_onsets_array[sleep_indices[0]] - buffer_seconds
    upper = epoch_onsets_array[sleep_indices[-1]] + config.epoch_seconds + buffer_seconds
    keep = (epoch_onsets_array >= lower) & (epoch_onsets_array < upper)
    return epoch_onsets_array[keep], stages_array[keep]


def _epoch_to_log_psd(epoch: np.ndarray, sampling_rate: float, config: SleepEDFConfig) -> np.ndarray:
    """Convert one signal epoch to a fixed-grid log-PSD function."""
    epoch = signal.detrend(np.asarray(epoch, dtype=np.float64), type="constant")
    n_per_segment = min(len(epoch), int(round(config.welch_segment_seconds * sampling_rate)))
    n_per_segment = max(8, n_per_segment)
    frequencies, power = signal.welch(
        epoch,
        fs=sampling_rate,
        window="hann",
        nperseg=n_per_segment,
        noverlap=n_per_segment // 2,
        detrend="constant",
        scaling="density",
    )
    valid = (frequencies >= config.frequency_min_hz) & (
        frequencies <= config.frequency_max_hz
    )
    if valid.sum() < 2:
        raise ValueError("The requested frequency range is unavailable at this sampling rate.")
    floor = max(np.finfo(np.float64).tiny, float(np.median(power[valid])) * 1e-12)
    log_power_db = 10.0 * np.log10(np.maximum(power[valid], floor))
    curve = np.interp(config.frequency_grid, frequencies[valid], log_power_db)
    return curve.astype(np.float32)


def read_functional_record(
    pair: RecordPair, config: SleepEDFConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one night as log-PSD curves, stage labels, and epoch onsets."""
    pyedflib = _load_pyedflib(config.data_root)
    hyp_start, ann_onsets, ann_durations, ann_descriptions = _read_annotations(
        pair.hypnogram_path, pyedflib
    )

    reader = pyedflib.EdfReader(str(pair.psg_path))
    try:
        labels = [label.strip() for label in reader.getSignalLabels()]
        if config.eeg_channel not in labels:
            raise ValueError(
                f"Channel {config.eeg_channel!r} not found in {pair.psg_path.name}; "
                f"available channels are {labels}."
            )
        channel_index = labels.index(config.eeg_channel)
        sampling_rate = float(reader.getSampleFrequency(channel_index))
        signal_samples = int(reader.getNSamples()[channel_index])
        epoch_samples = int(round(config.epoch_seconds * sampling_rate))
        psg_start = reader.getStartdatetime()
        epoch_onsets, stages = _expand_scored_epochs(
            psg_start,
            hyp_start,
            ann_onsets,
            ann_durations,
            ann_descriptions,
            config,
        )

        curves: List[np.ndarray] = []
        kept_stages: List[int] = []
        kept_onsets: List[float] = []
        for onset, stage in zip(epoch_onsets, stages):
            start_sample = int(round(float(onset) * sampling_rate))
            stop_sample = start_sample + epoch_samples
            if start_sample < 0 or stop_sample > signal_samples:
                continue
            epoch = reader.readSignal(
                channel_index,
                start=start_sample,
                n=epoch_samples,
                digital=False,
            )
            if len(epoch) != epoch_samples or not np.all(np.isfinite(epoch)):
                continue
            curves.append(_epoch_to_log_psd(epoch, sampling_rate, config))
            kept_stages.append(int(stage))
            kept_onsets.append(float(onset))
    finally:
        reader.close()

    if not curves:
        raise ValueError(f"No usable EEG epochs were extracted from {pair.psg_path.name}.")
    return (
        np.stack(curves, axis=0),
        np.asarray(kept_stages, dtype=np.int8),
        np.asarray(kept_onsets, dtype=np.float64),
    )


def _is_continuous(onsets: np.ndarray, start: int, length: int, epoch_seconds: int) -> bool:
    differences = np.diff(onsets[start : start + length])
    return len(differences) == length - 1 and bool(
        np.all(np.isclose(differences, epoch_seconds, atol=1e-3))
    )


def make_record_windows(
    curves: np.ndarray,
    stages: np.ndarray,
    onsets: np.ndarray,
    pair: RecordPair,
    config: SleepEDFConfig,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    """Scan all windows uniformly and assign strict binary change labels."""
    window_length = config.window_length
    margin = config.min_change_margin
    records: List[Dict[str, object]] = []
    positive_records: List[Dict[str, object]] = []
    negative_candidates: List[Dict[str, object]] = []

    # Positive and negative candidates are generated by exactly the same
    # sliding-window scan and are labelled only after the window is cut.
    for start in range(0, len(stages) - window_length + 1, config.window_stride):
        if not _is_continuous(onsets, start, window_length, config.epoch_seconds):
            continue
        local_stages = stages[start : start + window_length]
        local_changes = np.flatnonzero(local_stages[1:] != local_stages[:-1]) + 1

        if len(local_changes) == 0:
            stage_name = STAGE_NAME[int(local_stages[0])]
            negative_candidates.append(
                {
                    "curve": curves[start : start + window_length],
                    "label": 0,
                    "change_location": -1,
                    "change_locations": "",
                    "number_of_changes": 0,
                    "subject": pair.subject,
                    "night": pair.night,
                    "start_second": float(onsets[start]),
                    "stage_from": stage_name,
                    "stage_to": stage_name,
                }
            )
            continue

        # Discard a positive window if any of its transitions is too close to
        # either edge. All remaining windows with one or more transitions are
        # retained as positives.
        if np.any(local_changes < margin) or np.any(
            local_changes > window_length - margin
        ):
            continue
        change_location = int(local_changes[0])
        positive_records.append(
            {
                "curve": curves[start : start + window_length],
                "label": 1,
                # Retain the first transition for backward compatibility with
                # existing single-location code, and store the complete set for
                # any future localization analysis.
                "change_location": change_location,
                "change_locations": ";".join(str(int(x)) for x in local_changes),
                "number_of_changes": int(len(local_changes)),
                "subject": pair.subject,
                "night": pair.night,
                "start_second": float(onsets[start]),
                "stage_from": STAGE_NAME[int(local_stages[change_location - 1])],
                "stage_to": STAGE_NAME[int(local_stages[change_location])],
            }
        )

    if config.negative_to_positive_ratio is None:
        records.extend(negative_candidates)
    else:
        requested_negatives = int(
            round(len(positive_records) * config.negative_to_positive_ratio)
        )
        if requested_negatives > 0 and negative_candidates:
            count = min(requested_negatives, len(negative_candidates))
            indices = rng.choice(len(negative_candidates), size=count, replace=False)
            records.extend(negative_candidates[int(i)] for i in indices)
    records.extend(positive_records)
    rng.shuffle(records)
    return records


def _stack_split(records: Sequence[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("A dataset split contains no windows.")
    x = np.stack([np.asarray(record["curve"], dtype=np.float32) for record in records])
    y = np.asarray([int(record["label"]) for record in records], dtype=np.float32)
    return x, y


def _metadata(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    keys = [
        "label",
        "change_location",
        "change_locations",
        "number_of_changes",
        "subject",
        "night",
        "start_second",
        "stage_from",
        "stage_to",
    ]
    return [{key: record[key] for key in keys} for record in records]


def _write_metadata_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_and_save(
    config: SleepEDFConfig,
    max_records: int | None = None,
    output_name: str | None = None,
    metadata_prefix: str = "sleep_edf",
    verbose: bool = True,
    nights: str = "all",
) -> Tuple[Path, Dict[str, object]]:
    """Build subject-level splits, normalize on training data, and save tensors."""
    config.validate()
    pairs = select_record_pairs(discover_record_pairs(config.raw_dir), nights)
    if max_records is not None:
        pairs = pairs[:max_records]
    subject_splits = split_subjects([pair.subject for pair in pairs], config)
    split_lookup = {
        subject: split_name
        for split_name, subjects in subject_splits.items()
        for subject in subjects
    }
    records_by_split: Dict[str, List[Dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    rng = np.random.default_rng(config.seed)

    for index, pair in enumerate(pairs, start=1):
        curves, stages, onsets = read_functional_record(pair, config)
        windows = make_record_windows(curves, stages, onsets, pair, config, rng)
        split_name = split_lookup[pair.subject]
        records_by_split[split_name].extend(windows)
        if verbose:
            positives = sum(int(window["label"]) for window in windows)
            print(
                f"[{index:02d}/{len(pairs):02d}] {pair.night}: "
                f"{len(curves)} epochs, {len(windows)} windows "
                f"({positives} positive) -> {split_name}"
            )

    arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        split_name: _stack_split(records)
        for split_name, records in records_by_split.items()
    }
    train_x = arrays["train"][0]
    normalization_mean = train_x.mean(axis=(0, 1), dtype=np.float64)
    normalization_std = train_x.std(axis=(0, 1), dtype=np.float64)
    normalization_std = np.maximum(normalization_std, 1e-6)

    normalized: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for split_name, (x, y) in arrays.items():
        x = (x - normalization_mean[None, None, :]) / normalization_std[None, None, :]
        x = np.clip(x, -config.normalization_clip, config.normalization_clip).astype(
            np.float32
        )
        normalized[split_name] = (x, y)

    import torch

    payload: Dict[str, object] = {
        "format_version": 2,
        "night_selection": nights,
        "frequency_grid_hz": torch.from_numpy(config.frequency_grid.astype(np.float32)),
        "normalization_mean": torch.from_numpy(normalization_mean.astype(np.float32)),
        "normalization_std": torch.from_numpy(normalization_std.astype(np.float32)),
        "subject_splits": subject_splits,
        "config": {**asdict(config), "data_root": "."},
    }
    key_names = {"train": "train", "validation": "val", "test": "test"}
    for split_name, short_name in key_names.items():
        x, y = normalized[split_name]
        # Convert N x T x J to the original T x J x N storage layout.
        payload[f"X_{short_name}"] = torch.from_numpy(x).permute(1, 2, 0).contiguous()
        # Preserve unnormalized values for a new split without inverting clipping.
        raw_x = arrays[split_name][0]
        payload[f"raw_log_psd_{short_name}"] = (
            torch.from_numpy(raw_x).permute(1, 2, 0).contiguous()
        )
        payload[f"Y_{short_name}"] = torch.from_numpy(y)
        payload[f"cp_{short_name}"] = torch.tensor(
            [int(row["change_location"]) for row in records_by_split[split_name]],
            dtype=torch.int64,
        )
        payload[f"change_locations_{short_name}"] = [
            tuple(
                int(location)
                for location in str(row["change_locations"]).split(";")
                if location
            )
            for row in records_by_split[split_name]
        ]
        payload[f"metadata_{short_name}"] = _metadata(records_by_split[split_name])

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"sleep_edf_logpsd_T{config.window_length}.pt"
    output_path = config.processed_dir / output_name
    torch.save(payload, output_path)

    for split_name, short_name in key_names.items():
        _write_metadata_csv(
            config.processed_dir / f"{metadata_prefix}_{short_name}_metadata.csv",
            payload[f"metadata_{short_name}"],
        )
    subject_splits_name = (
        "subject_splits.json"
        if metadata_prefix == "sleep_edf"
        else f"{metadata_prefix}_subject_splits.json"
    )
    with (config.processed_dir / subject_splits_name).open("w", encoding="utf-8") as handle:
        json.dump(subject_splits, handle, indent=2)
    return output_path, payload


def check_fenet_compatibility(payload: Dict[str, object]) -> Dict[str, Tuple[int, ...]]:
    """Check the tensor and legacy notebook embedding shapes."""
    import torch
    from simulation.fenet import gram_embedding

    x_train = payload["X_train"]
    y_train = payload["Y_train"]
    if not isinstance(x_train, torch.Tensor) or not isinstance(y_train, torch.Tensor):
        raise TypeError("The saved X_train and Y_train objects must be torch tensors.")
    t, _, n = x_train.shape
    if y_train.shape != (n,):
        raise AssertionError(f"Expected Y_train shape {(n,)}, received {tuple(y_train.shape)}.")

    z = x_train[:, :, 0]
    embedded = gram_embedding(z, convention="legacy-notebook")
    if embedded.numel() != t * (t - 1):
        raise AssertionError("Unexpected FENet embedding dimension.")
    return {
        "X_train": tuple(x_train.shape),
        "Y_train": tuple(y_train.shape),
        "one_FENet_feature": tuple(embedded.shape),
    }
