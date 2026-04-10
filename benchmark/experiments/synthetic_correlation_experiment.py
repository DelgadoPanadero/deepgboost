"""
Synthetic correlation experiment.

Hypothesis 1
    DGBF outperforms XGBoost and RandomForest when input features are highly
    correlated (mean pairwise |correlation| > 0.5).

Hypothesis 2
    DGBF has a small-n advantage over competitors at fixed high correlation.

Usage
-----
    cd /home/thinbaker/Workspace/DeepGBoost
    .venv/bin/python -m benchmark.experiments.synthetic_correlation_experiment
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from scipy.linalg import toeplitz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from deepgboost import DeepGBoostRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_BENCHMARK_DIR, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Model budget: n_layers × n_trees ≤ 100
# ---------------------------------------------------------------------------
DGBF_PARAMS: dict[str, Any] = {
    "n_layers": 5,
    "n_trees": 20,          # 5 × 20 = 100  (budget limit)
    "learning_rate": 0.8,
    "n_jobs": 8,
}
XGB_PARAMS: dict[str, Any] = {"n_estimators": 100, "random_state": RANDOM_SEED}
RF_PARAMS: dict[str, Any] = {"n_estimators": 100, "random_state": RANDOM_SEED, "n_jobs": -1}

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _toeplitz_covariance(p: int, rho: float) -> np.ndarray:
    """Return a p×p Toeplitz correlation matrix with off-diagonal decay rho^|i-j|."""
    first_row = rho ** np.arange(p)
    return toeplitz(first_row)


def _target(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Nonlinear target: sin(x0) + x1^2 + x2*x3 + Gaussian noise."""
    noise = rng.standard_normal(X.shape[0]) * 0.3
    return np.sin(X[:, 0]) + X[:, 1] ** 2 + X[:, 2] * X[:, 3] + noise


def make_correlated_dataset(
    n: int,
    p: int,
    rho: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw X ~ N(0, Sigma_toeplitz(rho)) and compute nonlinear y."""
    Sigma = _toeplitz_covariance(p, rho)
    # Cholesky for efficient sampling
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, p))
    X = Z @ L.T
    y = _target(X, rng)
    return X, y


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

def cross_val_r2(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Return (mean_r2, std_r2) over n_folds stratified folds."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_scores.append(float(r2_score(y_test, y_pred)))
    arr = np.array(fold_scores)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def _model_factories() -> dict[str, Any]:
    return {
        "DGBF": lambda: DeepGBoostRegressor(**DGBF_PARAMS),
        "XGBoost": lambda: XGBRegressor(**XGB_PARAMS),
        "RandomForest": lambda: RandomForestRegressor(**RF_PARAMS),
    }


def run_rho_sweep(
    rho_values: list[float],
    n: int,
    p: int,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """
    Hypothesis 1: sweep rho at fixed dataset size.

    Returns a list of result dicts, one per (rho, model) combination.
    """
    rng = np.random.default_rng(seed)
    factories = _model_factories()
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 72)
    print("Hypothesis 1 — rho sweep (n={}, p={}, {}-fold CV)".format(n, p, n_folds))
    print("=" * 72)
    header = f"{'rho':>6}  {'model':>14}  {'mean R²':>10}  {'std R²':>10}  {'time(s)':>8}"
    print(header)
    print("-" * len(header))

    for rho in rho_values:
        X, y = make_correlated_dataset(n, p, rho, rng)
        for model_name, factory in factories.items():
            t0 = time.perf_counter()
            mean_r2, std_r2 = cross_val_r2(factory, X, y, n_folds=n_folds, seed=seed)
            elapsed = time.perf_counter() - t0
            print(
                f"{rho:>6.1f}  {model_name:>14}  {mean_r2:>10.4f}  {std_r2:>10.4f}  {elapsed:>8.2f}"
            )
            results.append(
                {
                    "experiment": "rho_sweep",
                    "rho": rho,
                    "n": n,
                    "p": p,
                    "model": model_name,
                    "mean_r2": mean_r2,
                    "std_r2": std_r2,
                    "n_folds": n_folds,
                    "time_seconds": round(elapsed, 4),
                }
            )

    return results


def run_n_sweep(
    n_values: list[int],
    rho: float,
    p: int,
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """
    Hypothesis 2: sweep dataset size at fixed high correlation.

    Returns a list of result dicts, one per (n, model) combination.
    """
    rng = np.random.default_rng(seed)
    factories = _model_factories()
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 72)
    print("Hypothesis 2 — n sweep (rho={}, p={}, {}-fold CV)".format(rho, p, n_folds))
    print("=" * 72)
    header = f"{'n':>7}  {'model':>14}  {'mean R²':>10}  {'std R²':>10}  {'time(s)':>8}"
    print(header)
    print("-" * len(header))

    for n in n_values:
        X, y = make_correlated_dataset(n, p, rho, rng)
        for model_name, factory in factories.items():
            t0 = time.perf_counter()
            mean_r2, std_r2 = cross_val_r2(factory, X, y, n_folds=n_folds, seed=seed)
            elapsed = time.perf_counter() - t0
            print(
                f"{n:>7}  {model_name:>14}  {mean_r2:>10.4f}  {std_r2:>10.4f}  {elapsed:>8.2f}"
            )
            results.append(
                {
                    "experiment": "n_sweep",
                    "rho": rho,
                    "n": n,
                    "p": p,
                    "model": model_name,
                    "mean_r2": mean_r2,
                    "std_r2": std_r2,
                    "n_folds": n_folds,
                    "time_seconds": round(elapsed, 4),
                }
            )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Hypothesis 1: correlation sweep ----
    rho_results = run_rho_sweep(
        rho_values=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
        n=2000,
        p=20,
        n_folds=5,
    )
    rho_output = os.path.join(_RESULTS_DIR, "synthetic_correlation_experiment.json")
    with open(rho_output, "w") as fh:
        json.dump(rho_results, fh, indent=2)
    print(f"\nRho-sweep results saved to: {rho_output}")

    # ---- Hypothesis 2: sample-size sweep ----
    n_results = run_n_sweep(
        n_values=[200, 500, 1000, 3000, 10000],
        rho=0.7,
        p=20,
        n_folds=5,
    )
    n_output = os.path.join(_RESULTS_DIR, "synthetic_n_experiment.json")
    with open(n_output, "w") as fh:
        json.dump(n_results, fh, indent=2)
    print(f"N-sweep results saved to: {n_output}")
