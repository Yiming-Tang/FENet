"""Generate independent simulations, train FENet, and save reviewable outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .fenet import (FENet, classification_metrics, detect_changepoints,
                   fit_classifier, gram_embedding, predict_probabilities, set_seed)
from .data import SimulationConfig, generate_dataset, generate_multiple_changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/simulation"))
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--innovations", choices=["IID", "FAR"], default="IID")
    parser.add_argument("--window-length", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--noise-sd", type=float, default=0.1)
    parser.add_argument("--test-strength", type=float, default=1.0)
    parser.add_argument("--decay", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PyTorch device, such as cpu, cuda, or mps.")
    parser.add_argument("--embedding", choices=["paper", "legacy-notebook"], default="paper")
    parser.add_argument("--multiple-length", type=int, default=0, help="Also localize a generated two-change sequence when > 0.")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()
    config = SimulationConfig(window_length=args.window_length, grid_size=args.grid_size,
                              scenario=args.scenario, innovations=args.innovations,
                              noise_sd=args.noise_sd, decay=args.decay, test_strength=args.test_strength)
    train_x, train_y, _ = generate_dataset(args.n_train, config, seed=args.seed)
    test_x, test_y, test_locations = generate_dataset(args.n_test, config, seed=args.seed + 20000, training=False)
    train_features = gram_embedding(torch.from_numpy(train_x), args.embedding)
    test_features = gram_embedding(torch.from_numpy(test_x), args.embedding)
    set_seed(args.seed)
    model = FENet(args.window_length)
    history = fit_classifier(model, train_features, torch.from_numpy(train_y),
                             epochs=args.epochs, batch_size=args.batch_size,
                             lr=args.learning_rate, seed=args.seed, device=args.device)
    probabilities = predict_probabilities(model, test_features, device=args.device)
    metrics = classification_metrics(test_y, probabilities.numpy())
    json_metrics = {key: value if np.isfinite(value) else None for key, value in metrics.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {**vars(args), "output_dir": str(args.output_dir), "simulation": asdict(config)}
    (args.output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(json_metrics, indent=2) + "\n", encoding="utf-8")
    np.savetxt(args.output_dir / "predictions.csv",
               np.column_stack([test_y, probabilities.numpy(), test_locations]), delimiter=",",
               header="label,probability,first_changed_index", comments="")
    torch.save({"state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
                "window_length": args.window_length, "hidden_dims": [256, 64],
                "embedding": args.embedding, "config": settings}, args.output_dir / "model.pt")
    if args.multiple_length:
        x, locations = generate_multiple_changes(args.multiple_length, config, seed=args.seed + 30000)
        detected = detect_changepoints(model, torch.from_numpy(x), gamma=0.5,
                                      smooth_window=args.smooth_window, convention=args.embedding,
                                      device=args.device)
        detected["true_first_changed_indices"] = locations
        (args.output_dir / "localization.json").write_text(json.dumps(detected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_metrics, indent=2))
    print(f"Saved experiment outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
