"""Supplementary Sleep-EDF benchmark with explicit subject and window splits.

The neural architectures and optimization settings are ported from the supplied
Sleep-EDF notebook. The MLP and legacy frequency CNN deliberately retain the
notebook's V-matrix convention.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# These defaults reproduce the final settings of the supplied sleep notebook.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.10
BATCH_SIZE = 64
MAX_EPOCHS = 40
MLP_EPOCHS = 30
PATIENCE = 7
KLCPD_WINDOW = 3
KLCPD_EPOCHS = 20
KLCPD_CRITIC_STEPS = 5
KLCPD_HIDDEN_DIM = 32
KLCPD_LR = 3e-4
KLCPD_WEIGHT_CLIP = 0.1
KLCPD_LAMBDA_AE = 1e-3
KLCPD_LAMBDA_REAL = 0.1
DEVICE = torch.device("cpu")

METHODS = {
    "fenet-mlp": "FENet (MLP)",
    "temporal-cnn": "FENet (CNN, temporal-CUSUM)",
    "legacy-cnn": "FENet (CNN, legacy)",
    "bigru": "BiGRU",
    "klcpd": "KL-CPD",
}

METHOD_TRANSFORMS = {
    "fenet-mlp": {
        "input": "training-standardized Gram features",
        "embedding_convention": "legacy-notebook",
    },
    "legacy-cnn": {
        "input": "raw and within-frequency V transform",
        "embedding_convention": "legacy-notebook",
    },
    "temporal-cnn": {"input": "raw, adjacent-difference, and temporal-CUSUM curves"},
    "bigru": {"input": "normalized functional curves"},
    "klcpd": {"input": "normalized functional curves; no-change training reference pool"},
}

def load_sleep_edf_pool(path: Path, allow_clipped_input: bool = False):
    """Pool saved windows before splitting and use original, unnormalized curves.

    Legacy files contain only clipped standardized curves. Their inverse is an
    approximation, so using them requires an explicit compatibility option.
    Only load trusted local PyTorch files: the format contains Python metadata.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    split_names = ("train", "val", "test")
    have_raw = all(f"raw_log_psd_{s}" in payload for s in split_names)
    if have_raw:
        x_log_psd = torch.cat([payload[f"raw_log_psd_{s}"].float() for s in split_names], dim=2)
    elif allow_clipped_input:
        import warnings
        warnings.warn(
            "Using legacy clipped values: inverse normalization cannot restore "
            "the original log-PSD tails. Rebuild with real_data.preprocess for raw input.",
            stacklevel=2,
        )
        x_standardized = torch.cat([payload[f"X_{s}"].float() for s in split_names], dim=2)
        old_mean = payload["normalization_mean"].float().view(1, -1, 1)
        old_std = payload["normalization_std"].float().view(1, -1, 1)
        x_log_psd = x_standardized * old_std + old_mean
    else:
        raise ValueError(
            "The input lacks raw_log_psd_* tensors. Rebuild with python -m "
            "real_data.preprocess, or explicitly select --allow-clipped-input to "
            "approximate the original notebook's legacy input."
        )
    y = torch.cat([payload[f"Y_{s}"].long() for s in split_names])
    cp = torch.cat([payload[f"cp_{s}"].long() for s in split_names])
    metadata = sum((list(payload[f"metadata_{s}"]) for s in split_names), [])
    if x_log_psd.ndim != 3 or x_log_psd.shape[2] != len(metadata) or y.shape != cp.shape:
        raise ValueError("Sleep-EDF tensors and metadata are misaligned.")
    if len(y) != len(metadata) or not torch.isfinite(x_log_psd).all():
        raise ValueError("Invalid Sleep-EDF values or metadata length.")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("The dataset must contain both binary classes.")
    return {
        "X_log_psd": x_log_psd.contiguous(),
        "Y": y.contiguous(),
        "cp": cp.contiguous(),
        "metadata": metadata,
        "subjects": np.asarray([row["subject"] for row in metadata]),
        "frequency_grid_hz": payload["frequency_grid_hz"].float(),
        "signal_channel": str(payload["config"]["eeg_channel"]),
        "normalization_clip": float(payload["config"]["normalization_clip"]),
        "input_values": "raw_log_psd" if have_raw else "inverse_clipped_legacy",
    }


def _materialize_split(pool, indices, split_seed, strategy):
    all_indices = np.arange(len(pool["Y"]))
    combined = np.concatenate(list(indices.values()))
    if not np.array_equal(np.sort(combined), all_indices):
        raise AssertionError("The split has overlap or missing indices.")
    train_index = torch.as_tensor(indices["train"], dtype=torch.long)
    train_x = pool["X_log_psd"].index_select(2, train_index)
    mean = train_x.mean(dim=(0, 2), keepdim=True)
    std = train_x.std(dim=(0, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
    x_normalized = ((pool["X_log_psd"] - mean) / std).clamp(
        -pool["normalization_clip"], pool["normalization_clip"]
    )
    split = {
        "split_seed": int(split_seed),
        "split_strategy": strategy,
        "indices": indices,
        "normalization_mean": mean.squeeze(0).squeeze(-1),
        "normalization_std": std.squeeze(0).squeeze(-1),
        "metadata_all": pool["metadata"],
        "frequency_grid_hz": pool["frequency_grid_hz"],
        "input_values": pool.get("input_values", "raw_log_psd"),
    }
    for name, idx in indices.items():
        tensor_idx = torch.as_tensor(idx, dtype=torch.long)
        split[f"X_{name}"] = x_normalized.index_select(2, tensor_idx).contiguous()
        split[f"Y_{name}"] = pool["Y"].index_select(0, tensor_idx).contiguous()
        split[f"cp_{name}"] = pool["cp"].index_select(0, tensor_idx).contiguous()
        if set(split[f"Y_{name}"].tolist()) != {0, 1}:
            raise ValueError(f"The {name} split must contain both labels; select another seed.")
    return split


def make_sample_split(pool, split_seed):
    """Reproduce the supplied stratified 70/10/20 window-level split protocol.

    Subject and recording overlap across splits is allowed by this protocol.
    It evaluates held-out windows rather than unseen-subject generalization.
    """
    all_indices = np.arange(len(pool["Y"]))
    labels = pool["Y"].cpu().numpy()
    n_total = len(all_indices)
    n_test = int(round((1.0 - TRAIN_FRACTION - VALIDATION_FRACTION) * n_total))
    n_val = int(round(VALIDATION_FRACTION * n_total))
    train_val, test = train_test_split(
        all_indices, test_size=n_test, random_state=split_seed,
        shuffle=True, stratify=labels,
    )
    train, val = train_test_split(
        train_val, test_size=n_val, random_state=split_seed + 1,
        shuffle=True, stratify=labels[train_val],
    )
    indices = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    return _materialize_split(pool, indices, split_seed, "sample")


def make_subject_split(pool, split_seed):
    """Create a subject-disjoint 60/20/20 split, matching preprocessing ratios."""
    from .preprocessing import SleepEDFConfig, split_subjects
    config = SleepEDFConfig(data_root=Path("."), seed=split_seed)
    subjects = split_subjects(pool["subjects"], config)
    indices = {
        ("val" if name == "validation" else name): np.flatnonzero(
            np.isin(pool["subjects"], members)
        )
        for name, members in subjects.items()
    }
    return _materialize_split(pool, indices, split_seed, "subject")


def export_split_for_r(split, output_root):
    split_dir = output_root / f"split_seed_{split['split_seed']}"
    r_dir = split_dir / 'r_test'
    r_dir.mkdir(parents=True, exist_ok=True)

    x_test = split['X_test'].cpu().numpy()  # T x J x N_test
    y_test = split['Y_test'].cpu().numpy().astype(int)
    cp_test = split['cp_test'].cpu().numpy().astype(int)
    test_global_indices = split['indices']['test']
    manifest_rows = []
    for local_index, global_index in enumerate(test_global_indices):
        sample_name = f'sample_{local_index + 1:04d}.csv'
        np.savetxt(r_dir / sample_name, x_test[:, :, local_index], delimiter=',', fmt='%.9g')
        row = split['metadata_all'][int(global_index)]
        manifest_rows.append({
            'sample_file': sample_name,
            'global_index': int(global_index),
            'split_seed': split['split_seed'],
            'label': int(y_test[local_index]),
            'change_location': int(cp_test[local_index]),
            'subject': row['subject'],
            'night': row['night'],
            'start_second': row['start_second'],
            'stage_from': row['stage_from'],
            'stage_to': row['stage_to'],
        })
    np.savetxt(r_dir / 'Y.csv', y_test[:, None], delimiter=',', fmt='%d')
    np.savetxt(r_dir / 'change_location.csv', cp_test[:, None], delimiter=',', fmt='%d')
    np.savetxt(
        r_dir / 'frequency_grid_hz.csv',
        split['frequency_grid_hz'].cpu().numpy()[:, None],
        delimiter=',', fmt='%.9g',
    )
    pd.DataFrame(manifest_rows).to_csv(r_dir / 'manifest.csv', index=False)

    all_manifest_rows = []
    for split_name in ('train', 'val', 'test'):
        for global_index in split['indices'][split_name]:
            row = split['metadata_all'][int(global_index)]
            all_manifest_rows.append({
                'global_index': int(global_index),
                'split_seed': split['split_seed'],
                'split': split_name,
                'subject': row['subject'],
                'label': int(row['label']),
                'change_location': int(row['change_location']),
            })
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / 'protocol.json').write_text(
        json.dumps({key: split[key] for key in ('split_seed', 'split_strategy', 'input_values')}, indent=2) + '\n'
    )
    pd.DataFrame(all_manifest_rows).sort_values('global_index').to_csv(
        split_dir / 'split_manifest.csv', index=False
    )
    torch.save(
        {
            'mean': split['normalization_mean'],
            'std': split['normalization_std'],
        },
        split_dir / 'training_normalization.pt',
    )
    return r_dir


def build_v_matrix(length):
    """Use the original sleep notebook convention explicitly."""
    from simulation.fenet import build_v_matrix as shared_build_v_matrix
    return shared_build_v_matrix(length, convention="legacy-notebook")


class FENetMLP(nn.Module):
    def __init__(self, T, hidden1=256, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(T * (T - 1)),
            nn.Linear(T * (T - 1), hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def temporal_cusum_channels(x):
    '''Raw, adjacent-difference, and temporal FCUSUM curves.

    Parameters
    ----------
    x : tensor, shape (N, T, J)
        A batch of functional time series.
    '''
    _, T, _ = x.shape
    t = torch.arange(1, T, dtype=x.dtype, device=x.device).view(1, -1, 1)
    partial = x.cumsum(dim=1)[:, :-1, :]
    total = x.sum(dim=1, keepdim=True)
    cusum = (partial - (t / T) * total) / np.sqrt(T)
    adjacent_difference = x[:, 1:, :] - x[:, :-1, :]
    return torch.cat([x, adjacent_difference, cusum], dim=1)


class FENetLegacyCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FENetTemporalCUSUMCNN(nn.Module):
    '''Frequency CNN with an explicit temporal-FCUSUM comparator branch.'''

    def __init__(self, T):
        super().__init__()
        self.T = T
        channels = 3 * T - 2
        self.spectral = nn.Sequential(
            nn.Conv1d(channels, 48, kernel_size=7, padding=3),
            nn.GroupNorm(8, 48), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64), nn.ReLU(),
        )
        self.cusum_branch = nn.Sequential(
            nn.LayerNorm(T - 1),
            nn.Linear(T - 1, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 + 64 + 32 + 1, 1),
        )

    def forward(self, x):
        spectral_map = self.spectral(x)
        spectral_max = F.adaptive_max_pool1d(spectral_map, 1).squeeze(-1)
        spectral_mean = F.adaptive_avg_pool1d(spectral_map, 1).squeeze(-1)

        # Channel order: raw T, differences T-1, temporal CUSUM T-1.
        cusum_curves = x[:, 2 * self.T - 1:, :]
        m_values = torch.log1p(cusum_curves.square().mean(dim=-1))
        m_features = self.cusum_branch(m_values)
        max_m = m_values.max(dim=1, keepdim=True).values
        features = torch.cat(
            [spectral_max, spectral_mean, m_features, max_m], dim=1
        )
        return self.head(features).squeeze(-1)


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, emb_dim=32):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, emb_dim)

    def forward(self, x):
        hidden, _ = self.rnn(x)
        return F.normalize(self.proj(hidden.mean(dim=1)), dim=-1)


class BiGRUDiscrepancyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, emb_dim=32):
        super().__init__()
        self.encoder = SequenceEncoder(input_dim, hidden_dim, emb_dim)
        self.cls = nn.Sequential(
            nn.Linear(emb_dim * 3, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        midpoint = x.shape[1] // 2
        z_left = self.encoder(x[:, :midpoint, :])
        z_right = self.encoder(x[:, midpoint:, :])
        features = torch.cat([z_left, z_right, torch.abs(z_left - z_right)], dim=-1)
        return self.cls(features).squeeze(-1)


class KLCPDGenerator(nn.Module):
    '''Conditional recurrent generator corresponding to NetG in KL-CPD.'''

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    @staticmethod
    def shift_right(x):
        shifted = torch.zeros_like(x)
        shifted[:, 1:, :] = x[:, :-1, :]
        return shifted

    def forward(self, past, future, noise):
        _, hidden = self.encoder(past)
        decoder_input = self.shift_right(future)
        generated, _ = self.decoder(
            decoder_input, hidden + noise.unsqueeze(0)
        )
        return self.output(generated)


class KLCPDDeepKernel(nn.Module):
    '''Recurrent encoder-decoder corresponding to NetD in KL-CPD.'''

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return encoded, decoded


def pairwise_squared_distance(x, y):
    # x: B x n x d; y: B x m x d
    distance = (
        x.square().sum(dim=-1, keepdim=True)
        + y.square().sum(dim=-1).unsqueeze(1)
        - 2.0 * torch.bmm(x, y.transpose(1, 2))
    )
    return distance.clamp_min(0.0)


def batch_rbf_mmd2(x, y, sigmas):
    '''Biased per-window MMD^2 with the KL-CPD mixture RBF kernel.'''
    d_xx = pairwise_squared_distance(x, x)
    d_yy = pairwise_squared_distance(y, y)
    d_xy = pairwise_squared_distance(x, y)
    scale = 2.0 * sigmas.square().view(-1, 1, 1, 1)
    k_xx = torch.exp(-d_xx.unsqueeze(0) / scale).mean(dim=0)
    k_yy = torch.exp(-d_yy.unsqueeze(0) / scale).mean(dim=0)
    k_xy = torch.exp(-d_xy.unsqueeze(0) / scale).mean(dim=0)
    return k_xx.mean(dim=(1, 2)) + k_yy.mean(dim=(1, 2)) - 2.0 * k_xy.mean(dim=(1, 2))


def prepare_method_tensors(method, split):
    T, J, _ = split['X_train'].shape
    tensors = {}
    if method == 'FENet (MLP)':
        v_time = build_v_matrix(T)
        embedded = {}
        for name in ('train', 'val', 'test'):
            z = split[f'X_{name}'].permute(2, 0, 1).float()  # N x T x J
            gram = torch.bmm(z, z.transpose(1, 2))
            embedded[name] = torch.matmul(gram, v_time).reshape(len(z), -1)
        # Standardize every embedding coordinate using training samples
        # only, then apply the same transformation to validation/test.
        embedding_mean = embedded['train'].mean(dim=0, keepdim=True)
        embedding_std = embedded['train'].std(
            dim=0, unbiased=False, keepdim=True
        ).clamp_min(1e-6)
        for name in ('train', 'val', 'test'):
            tensors[name] = (
                embedded[name] - embedding_mean
            ) / embedding_std
        model_factory = lambda: FENetMLP(T)
        optimizer_settings = {'lr': 3e-4, 'weight_decay': 1e-4}

    elif method == 'FENet (CNN, legacy)':
        train_x = split['X_train'].float()
        channel_mean = train_x.mean(dim=(1, 2), keepdim=True)
        channel_std = train_x.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-6)
        v_grid = build_v_matrix(J)
        for name in ('train', 'val', 'test'):
            normalized = (split[f'X_{name}'].float() - channel_mean) / channel_std
            raw = normalized.permute(2, 0, 1)  # N x T x J
            transformed = torch.matmul(raw, v_grid)  # N x T x (J-1)
            tensors[name] = torch.cat([raw[:, :, 1:], transformed], dim=1)
        model_factory = lambda: FENetLegacyCNN(2 * T)
        optimizer_settings = {'lr': 1e-3, 'weight_decay': 1e-3}

    elif method == 'FENet (CNN, temporal-CUSUM)':
        for name in ('train', 'val', 'test'):
            raw = split[f'X_{name}'].permute(2, 0, 1).float()
            tensors[name] = temporal_cusum_channels(raw)
        model_factory = lambda: FENetTemporalCUSUMCNN(T)
        optimizer_settings = {'lr': 3e-4, 'weight_decay': 1e-4}

    elif method == 'BiGRU':
        for name in ('train', 'val', 'test'):
            tensors[name] = split[f'X_{name}'].permute(2, 0, 1).float()
        model_factory = lambda: BiGRUDiscrepancyClassifier(J)
        optimizer_settings = {'lr': 1e-3, 'weight_decay': 0.0}
    else:
        raise ValueError(f'Unknown method: {method}')

    labels = {name: split[f'Y_{name}'].float() for name in ('train', 'val', 'test')}
    return tensors, labels, model_factory, optimizer_settings

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loaders(tensors, labels, seed):
    loaders = {}
    for name in ('train', 'val', 'test'):
        dataset = TensorDataset(tensors[name], labels[name])
        generator = torch.Generator().manual_seed(seed) if name == 'train' else None
        loaders[name] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(name == 'train'),
            generator=generator,
        )
    return loaders


def epoch_loss(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def collect_scores(model, loader):
    model.eval()
    scores, labels = [], []
    for xb, yb in loader:
        scores.append(torch.sigmoid(model(xb.to(DEVICE))).cpu())
        labels.append(yb.long().cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def validation_threshold(y_true, y_score):
    # KL-CPD returns an MMD score rather than a probability, so a fixed
    # [0, 1] grid is not a method-neutral threshold search. Every method
    # instead uses its observed validation-score cut points. Ties in F1
    # are resolved by validation accuracy and then the higher threshold.
    candidates = np.unique(np.asarray(y_score, dtype=float))
    records = []
    for threshold in candidates:
        prediction = y_score >= threshold
        records.append((
            f1_score(y_true, prediction, zero_division=0),
            accuracy_score(y_true, prediction),
            float(threshold),
        ))
    best_f1, _, best_threshold = max(records)
    return best_threshold, float(best_f1)


def classification_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'TPR': tp / (tp + fn) if tp + fn else 0.0,
        'FAR': fp / (fp + tn) if fp + tn else 0.0,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_score),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    }


def sample_klcpd_reference_pairs(x, window):
    '''Sample equal past/future windows from stationary reference series.'''
    batch_size, length, _ = x.shape
    if 2 * window > length:
        raise ValueError('KL-CPD window is too long for the input sequence.')
    centers = torch.randint(
        window, length - window + 1, (batch_size,), device=x.device
    )
    rows = torch.arange(batch_size, device=x.device).unsqueeze(1)
    past_offsets = torch.arange(-window, 0, device=x.device).unsqueeze(0)
    future_offsets = torch.arange(0, window, device=x.device).unsqueeze(0)
    past = x[rows, centers.unsqueeze(1) + past_offsets]
    future = x[rows, centers.unsqueeze(1) + future_offsets]
    return past, future


@torch.no_grad()
def initialize_klcpd_sigmas(deep_kernel, reference_x, window):
    subset = reference_x[:min(128, len(reference_x))].to(DEVICE)
    past, future = sample_klcpd_reference_pairs(subset, window)
    past_encoded, _ = deep_kernel(past)
    future_encoded, _ = deep_kernel(future)
    encoded = torch.cat([past_encoded, future_encoded], dim=1).reshape(
        -1, past_encoded.shape[-1]
    )
    distances = torch.pdist(encoded.float().cpu())
    positive = distances[distances > 0]
    median = positive.median().item() if len(positive) else 1.0
    multipliers = torch.tensor(
        [0.25, 0.5, 1.0, 2.0, 4.0], device=DEVICE
    )
    return (median * multipliers).clamp_min(1e-3)


@torch.no_grad()
def klcpd_normal_mmd(deep_kernel, reference_x, sigmas, window):
    deep_kernel.eval()
    values = []
    for start in range(0, len(reference_x), BATCH_SIZE):
        xb = reference_x[start:start + BATCH_SIZE].to(DEVICE)
        # A fixed midpoint makes checkpoint selection deterministic and
        # does not use change/no-change validation labels.
        midpoint = xb.shape[1] // 2
        past = xb[:, midpoint - window:midpoint, :]
        future = xb[:, midpoint:midpoint + window, :]
        past_encoded, _ = deep_kernel(past)
        future_encoded, _ = deep_kernel(future)
        values.append(batch_rbf_mmd2(
            past_encoded, future_encoded, sigmas
        ).cpu())
    return float(torch.cat(values).mean())


@torch.no_grad()
def collect_klcpd_scores(deep_kernel, x, sigmas, window):
    '''Maximum native deep-kernel MMD across admissible boundaries.'''
    deep_kernel.eval()
    all_scores = []
    for start in range(0, len(x), BATCH_SIZE):
        xb = x[start:start + BATCH_SIZE].to(DEVICE)
        candidate_scores = []
        for boundary in range(window, xb.shape[1] - window + 1):
            past = xb[:, boundary - window:boundary, :]
            future = xb[:, boundary:boundary + window, :]
            past_encoded, _ = deep_kernel(past)
            future_encoded, _ = deep_kernel(future)
            candidate_scores.append(batch_rbf_mmd2(
                past_encoded, future_encoded, sigmas
            ))
        all_scores.append(torch.stack(candidate_scores, dim=1).max(dim=1).values.cpu())
    return torch.cat(all_scores).numpy()


def run_one_klcpd(split, model_seed, verbose=True):
    '''Modern-PyTorch adaptation of the official KL-CPD objective.'''
    set_all_seeds(model_seed)
    T, J, _ = split['X_train'].shape
    if 2 * KLCPD_WINDOW > T:
        raise ValueError('KLCPD_WINDOW is incompatible with T.')

    train_x = split['X_train'].permute(2, 0, 1).float()
    train_y = split['Y_train'].long()
    reference_x = train_x[train_y == 0].contiguous()
    if len(reference_x) < 2:
        raise ValueError('KL-CPD needs at least two no-change training windows.')
    loader_generator = torch.Generator().manual_seed(model_seed)
    reference_loader = DataLoader(
        TensorDataset(reference_x),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
    )

    generator = KLCPDGenerator(J, KLCPD_HIDDEN_DIM).to(DEVICE)
    deep_kernel = KLCPDDeepKernel(J, KLCPD_HIDDEN_DIM).to(DEVICE)
    optimizer_g = optim.Adam(generator.parameters(), lr=KLCPD_LR)
    optimizer_d = optim.Adam(deep_kernel.parameters(), lr=KLCPD_LR)
    sigmas = initialize_klcpd_sigmas(
        deep_kernel, reference_x, KLCPD_WINDOW
    )

    best_state = None
    best_epoch = 0
    best_normal_mmd = float('inf')
    for epoch in range(1, KLCPD_EPOCHS + 1):
        generator.train()
        deep_kernel.train()
        d_values, g_values = [], []
        for (xb,) in reference_loader:
            xb = xb.to(DEVICE)
            past, future = sample_klcpd_reference_pairs(
                xb, KLCPD_WINDOW
            )

            for _ in range(KLCPD_CRITIC_STEPS):
                for parameter in deep_kernel.parameters():
                    parameter.requires_grad_(True)
                noise = torch.randn(
                    len(xb), KLCPD_HIDDEN_DIM, device=DEVICE
                )
                with torch.no_grad():
                    generated = generator(past, future, noise)
                past_encoded, past_decoded = deep_kernel(past)
                future_encoded, future_decoded = deep_kernel(future)
                generated_encoded, generated_decoded = deep_kernel(generated)
                fake_mmd = batch_rbf_mmd2(
                    future_encoded, generated_encoded, sigmas
                ).mean()
                real_mmd = batch_rbf_mmd2(
                    past_encoded, future_encoded, sigmas
                ).mean()
                reconstruction = (
                    F.mse_loss(past_decoded, past)
                    + F.mse_loss(future_decoded, future)
                    + F.mse_loss(generated_decoded, generated)
                )
                # The deep kernel maximizes surrogate MMD while keeping
                # adjacent stationary windows close and reconstructable.
                loss_d = (
                    -fake_mmd
                    + KLCPD_LAMBDA_REAL * real_mmd
                    + KLCPD_LAMBDA_AE * reconstruction
                )
                optimizer_d.zero_grad()
                loss_d.backward()
                nn.utils.clip_grad_norm_(deep_kernel.parameters(), 10.0)
                optimizer_d.step()
                for parameter in deep_kernel.encoder.parameters():
                    parameter.data.clamp_(
                        -KLCPD_WEIGHT_CLIP, KLCPD_WEIGHT_CLIP
                    )
                d_values.append(float(loss_d.detach()))

            for parameter in deep_kernel.parameters():
                parameter.requires_grad_(False)
            noise = torch.randn(
                len(xb), KLCPD_HIDDEN_DIM, device=DEVICE
            )
            generated = generator(past, future, noise)
            future_encoded, _ = deep_kernel(future)
            generated_encoded, _ = deep_kernel(generated)
            loss_g = batch_rbf_mmd2(
                future_encoded, generated_encoded, sigmas
            ).mean()
            optimizer_g.zero_grad()
            loss_g.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
            optimizer_g.step()
            g_values.append(float(loss_g.detach()))
            for parameter in deep_kernel.parameters():
                parameter.requires_grad_(True)

        normal_mmd = klcpd_normal_mmd(
            deep_kernel, reference_x, sigmas, KLCPD_WINDOW
        )
        if normal_mmd < best_normal_mmd:
            best_normal_mmd = normal_mmd
            best_state = copy.deepcopy(deep_kernel.state_dict())
            best_epoch = epoch
        if verbose:
            print(
                f"KL-CPD | split {split['split_seed']} | seed {model_seed} | "
                f"E{epoch:02d} D {np.mean(d_values):.4f} "
                f"G {np.mean(g_values):.4f} normal-MMD {normal_mmd:.4f}"
            )

    deep_kernel.load_state_dict(best_state)
    val_x = split['X_val'].permute(2, 0, 1).float()
    test_x = split['X_test'].permute(2, 0, 1).float()
    val_true = split['Y_val'].long().numpy()
    test_true = split['Y_test'].long().numpy()
    val_score = collect_klcpd_scores(
        deep_kernel, val_x, sigmas, KLCPD_WINDOW
    )
    threshold, best_val_f1 = validation_threshold(val_true, val_score)
    test_score = collect_klcpd_scores(
        deep_kernel, test_x, sigmas, KLCPD_WINDOW
    )
    metrics = classification_metrics(test_true, test_score, threshold)
    return {
        'method': 'KL-CPD',
        'split_seed': split['split_seed'],
        'model_seed': model_seed,
        'best_epoch': best_epoch,
        'validation_threshold': threshold,
        'validation_F1': best_val_f1,
        'n_train': len(train_x),
        'n_reference_train': len(reference_x),
        'n_val': len(val_x),
        'n_test': len(test_x),
        **metrics,
    }


def run_one_method(method, split, model_seed, verbose=True):
    if method == 'KL-CPD':
        return run_one_klcpd(split, model_seed, verbose=verbose)

    set_all_seeds(model_seed)
    tensors, labels, model_factory, optimizer_settings = prepare_method_tensors(method, split)
    loaders = make_loaders(tensors, labels, model_seed)
    model = model_factory().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), **optimizer_settings)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    stale = 0
    epochs_to_run = MLP_EPOCHS if method == 'FENet (MLP)' else MAX_EPOCHS
    use_early_stopping = method != 'FENet (MLP)'
    for epoch in range(1, epochs_to_run + 1):
        train_loss = epoch_loss(model, loaders['train'], optimizer)
        val_loss = epoch_loss(model, loaders['val'])
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if verbose:
            print(
                f"{method} | split {split['split_seed']} | seed {model_seed} | "
                f"E{epoch:02d} train {train_loss:.4f} val {val_loss:.4f}"
            )
        if use_early_stopping and stale >= PATIENCE:
            break

    model.load_state_dict(best_state)
    val_score, val_true = collect_scores(model, loaders['val'])
    threshold, best_val_f1 = validation_threshold(val_true, val_score)
    test_score, test_true = collect_scores(model, loaders['test'])
    metrics = classification_metrics(test_true, test_score, threshold)
    result = {
        'method': method,
        'split_seed': split['split_seed'],
        'model_seed': model_seed,
        'best_epoch': best_epoch,
        'validation_threshold': threshold,
        'validation_F1': best_val_f1,
        'n_train': len(labels['train']),
        'n_val': len(labels['val']),
        'n_test': len(labels['test']),
        **metrics,
    }
    return result



def run_benchmark(
    input_path: Path,
    output_root: Path,
    split_strategy: str = "subject",
    split_seeds=(2026,),
    model_seeds=(101,),
    methods=("fenet-mlp", "temporal-cnn", "klcpd"),
    export_r: bool = False,
    export_only: bool = False,
    allow_clipped_input: bool = False,
):
    """Train paired comparisons and save per-run, per-split, and summary CSVs."""
    if split_strategy not in {"sample", "subject"}:
        raise ValueError("split_strategy must be 'sample' or 'subject'.")
    if not split_seeds or not model_seeds or not methods:
        raise ValueError("Provide at least one split seed, model seed, and method.")
    if set(methods) - set(METHODS):
        raise ValueError(f"Unknown methods: {set(methods) - set(METHODS)}")
    pool = load_sleep_edf_pool(input_path, allow_clipped_input=allow_clipped_input)
    output_root = Path(output_root) / split_strategy
    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "input_name": Path(input_path).name,
        "input_values": pool["input_values"],
        "split_strategy": split_strategy,
        "split_seeds": list(split_seeds),
        "model_seeds": list(model_seeds),
        "methods": list(methods),
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "epochs": {"mlp": MLP_EPOCHS, "other_supervised": MAX_EPOCHS, "klcpd": KLCPD_EPOCHS},
        "method_transforms": {name: METHOD_TRANSFORMS[name] for name in methods},
        "signal_channel": pool["signal_channel"],
    }
    (output_root / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    splitter = make_subject_split if split_strategy == "subject" else make_sample_split
    results = []
    for split_seed in split_seeds:
        split = splitter(pool, split_seed)
        print(f"Split {split_seed} ({split_strategy}): " + ", ".join(
            f"{name}={len(split[f'Y_{name}'])}" for name in ("train", "val", "test")
        ))
        if export_r or export_only:
            print("R test inputs:", export_split_for_r(split, output_root))
        if export_only:
            continue
        for model_seed in model_seeds:
            for short_name in methods:
                result = run_one_method(METHODS[short_name], split, model_seed)
                result["split_strategy"] = split_strategy
                results.append(result)
                pd.DataFrame(results).to_csv(output_root / "python_results.csv", index=False)
    results_df = pd.DataFrame(results)
    if results:
        metric_columns = ["Accuracy", "TPR", "FAR", "Precision", "F1", "AUC"]
        split_level = results_df.groupby(["method", "split_seed"], as_index=False)[metric_columns].mean()
        summary = split_level.groupby("method")[metric_columns].agg(["mean", "std", "count"])
        split_level.to_csv(output_root / "python_split_level_results.csv", index=False)
        summary.to_csv(output_root / "python_summary_mean_sd.csv")
        print(summary.round(4).to_string())
    return results_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sleep-edf/processed/sleep_edf_eog_two_nights_logpsd_T20.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sleep-edf"))
    parser.add_argument("--split-strategy", choices=("subject", "sample"), default="subject")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=["fenet-mlp", "temporal-cnn", "klcpd"])
    parser.add_argument("--quick", action="store_true", help="Use the first seeds and three training epochs.")
    parser.add_argument("--export-r", action="store_true")
    parser.add_argument("--export-only", action="store_true", help="Export paired test data without training.")
    parser.add_argument("--allow-clipped-input", action="store_true", help="Explicit compatibility mode for legacy clipped-only data.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    global DEVICE, MAX_EPOCHS, MLP_EPOCHS, PATIENCE, KLCPD_EPOCHS, KLCPD_CRITIC_STEPS
    if args.device == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        DEVICE = torch.device(args.device)
    if args.quick:
        MAX_EPOCHS = MLP_EPOCHS = PATIENCE = KLCPD_EPOCHS = 3
        KLCPD_CRITIC_STEPS = 2
        args.split_seeds = args.split_seeds[:1]
        args.model_seeds = args.model_seeds[:1]
    run_benchmark(
        args.input, args.output, args.split_strategy,
        args.split_seeds, args.model_seeds, args.methods,
        args.export_r, args.export_only, args.allow_clipped_input,
    )


if __name__ == "__main__":
    main()
