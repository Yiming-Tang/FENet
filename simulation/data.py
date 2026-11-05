"""Functional simulation designs consolidated from the supplied notebooks.

These are the numerical source-code conventions. In particular, change strength
``c`` enters as sqrt(c), and FAR dependence acts on sampled grid values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def basis_functions(x_values: np.ndarray, dimension: int) -> np.ndarray:
    """Use sqrt(2) sin(l*pi*x) for odd l and cos(l*pi*x) for even l."""
    order = np.arange(1, dimension + 1)
    angles = np.asarray(x_values)[:, None] * order[None, :] * np.pi
    return np.sqrt(2) * np.where(order % 2, np.sin(angles), np.cos(angles))


def make_basis(scenario: int, grid_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return the source basis: uniform/D21, Fourier/D6, or Fourier/D21."""
    if scenario not in {1, 2, 3} or grid_size < 2:
        raise ValueError("scenario must be 1, 2, or 3, with grid_size >= 2.")
    dimension = 6 if scenario == 2 else 21
    if scenario == 1:
        return rng.uniform(0, 1, (grid_size, dimension))
    return basis_functions(np.linspace(0, 1, grid_size), dimension)


def generate_noise(
    basis: np.ndarray, length: int, rng: np.random.Generator, *,
    innovations: str = "IID", noise_sd: float = 0.1, decay: int = 2,
    burn_in: int = 100,
) -> np.ndarray:
    """Generate source-style IID or grid-level FAR(1) residual curves.

    FAR uses a random grid_size square operator with Frobenius norm 0.5.
    The decay setting sets a scalar SD before normalization, as in the source;
    it does not change the normalized operator's distribution.
    """
    if length < 1 or noise_sd <= 0 or decay not in {1, 2, 3} or burn_in < 0:
        raise ValueError("Invalid noise-generation settings.")
    if innovations not in {"IID", "FAR"}:
        raise ValueError("innovations must be IID or FAR.")
    total = length + (burn_in if innovations == "FAR" else 0)
    dimension = basis.shape[1]
    coefficients = rng.normal(0, noise_sd, (total, dimension))
    noise = coefficients @ basis.T
    if innovations == "IID":
        return noise
    order = np.arange(1, dimension + 1)
    sigma = np.where(order <= 3, 1, 0) if decay == 1 else (3.0 ** -order if decay == 2 else 1 / order)
    operator = rng.normal(0, np.dot(sigma, sigma), (basis.shape[0], basis.shape[0]))
    operator *= 0.5 / np.linalg.norm(operator, "fro")
    residual = np.empty_like(noise)
    residual[0] = noise[0]
    for index in range(1, total):
        residual[index] = operator @ residual[index - 1] + noise[index]
    return residual[burn_in:]


def break_function(basis: np.ndarray, affected_dimension: int, strength: float) -> np.ndarray:
    """Return sqrt(c/xi) times the sum of the first xi basis columns."""
    if not 1 <= affected_dimension <= basis.shape[1] or strength < 0:
        raise ValueError("Invalid affected dimension or change strength.")
    return np.sqrt(strength / affected_dimension) * basis[:, :affected_dimension].sum(axis=1)


@dataclass(frozen=True)
class SimulationConfig:
    """Settings for independent, balanced single-change classification data."""

    window_length: int = 100
    grid_size: int = 50
    scenario: int = 3
    innovations: str = "IID"
    noise_sd: float = 0.1
    decay: int = 2
    train_strength_min: float = 0.1
    train_strength_max: float = 1.0
    test_strength: float = 1.0


def generate_dataset(
    sample_count: int, config: SimulationConfig, *, seed: int = 1,
    training: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N,T,J), binary labels, and zero-based first changed indices.

    Training positives sample c uniformly, xi uniformly from 1,...,D inclusive,
    and theta uniformly from [0.1,0.9]. Test positives use xi=D//2, theta=.5.
    The source Python condition i > floor(theta*T) is preserved. Negatives have
    location -1. Samples are shuffled together after exactly balanced labels
    are generated. Separate train/test seeds prevent source-notebook reuse.
    T must be at least 10 so theta < 0.9 always leaves a post-change curve.
    """
    if sample_count < 2 or sample_count % 2 or config.window_length < 10:
        raise ValueError("An even sample_count >= 2 and window_length >= 10 are required.")
    if not 0 < config.train_strength_min <= config.train_strength_max or config.test_strength <= 0:
        raise ValueError("Change strengths must be positive and ordered.")
    rng = np.random.default_rng(seed)
    template = make_basis(config.scenario, config.grid_size, rng)
    dimension = template.shape[1]
    labels = (np.arange(sample_count) % 2).astype(np.float32)
    x = np.empty((sample_count, config.window_length, config.grid_size), dtype=np.float32)
    locations = np.full(sample_count, -1, dtype=np.int64)
    for index, label in enumerate(labels):
        basis = template[:, rng.permutation(dimension)]
        curves = generate_noise(basis, config.window_length, rng,
                                innovations=config.innovations, noise_sd=config.noise_sd, decay=config.decay)
        if label:
            strength = rng.uniform(config.train_strength_min, config.train_strength_max) if training else config.test_strength
            affected = int(rng.integers(1, dimension + 1)) if training else dimension // 2
            theta = rng.uniform(0.1, 0.9) if training else 0.5
            location = int(np.floor(theta * config.window_length)) + 1
            curves[location:] += break_function(basis, affected, strength)
            locations[index] = location
        x[index] = curves
    order = rng.permutation(sample_count)
    return x[order], labels[order], locations[order]


def generate_multiple_changes(
    length: int, config: SimulationConfig, *, seed: int = 1,
    relative_locations: tuple[float, ...] = (0.3, 0.6),
    strengths: tuple[float, ...] = (1.0, 5.0),
) -> tuple[np.ndarray, list[int]]:
    """Add successive source-style break functions; return X (T*,J) and indices."""
    if not len(relative_locations) == len(strengths) or not relative_locations:
        raise ValueError("A matching nonempty location and strength list is required.")
    if any(not 0 < theta < 1 for theta in relative_locations) or sorted(relative_locations) != list(relative_locations):
        raise ValueError("Relative locations must be ordered and strictly inside (0, 1).")
    rng = np.random.default_rng(seed)
    basis = make_basis(config.scenario, config.grid_size, rng)
    basis = basis[:, rng.permutation(basis.shape[1])]
    curves = generate_noise(basis, length, rng, innovations=config.innovations,
                            noise_sd=config.noise_sd, decay=config.decay)
    locations = []
    for index, (theta, strength) in enumerate(zip(relative_locations, strengths), start=1):
        affected = max(1, index * basis.shape[1] // (len(strengths) + 1))
        location = int(np.floor(theta * length)) + 1
        if location >= length:
            raise ValueError("A change location leaves no post-change observation.")
        curves[location:] += break_function(basis, affected, strength)
        locations.append(location)
    return curves.astype(np.float32), locations
