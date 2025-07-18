# agent.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public helper functions already consumed by world.py – DO NOT BREAK THEM
# ---------------------------------------------------------------------------
class Agent:
    """Convenience namespace – simulation keeps per‑agent data in NumPy arrays."""

    RADIUS = 5.0  # for collision tests & rendering

    # -------------------------------------------------------------- spawning
    @staticmethod
    def random_spawn(n: int, w: float, h: float, initial_life: float = 100.0) -> Tuple[np.ndarray, ...]:
        """
        Return (xy, heading, speed, life) for `n` fresh agents.

        Shapes:
            xy       (n,2) float32, coords in [0,w)×[0,h)
            heading  (n,)  float32, radians [0,2π)
            speed    (n,)  float32, command ∈ [0,1]
            life     (n,)  float32
        """
        xy = np.empty((n, 2), np.float32)
        rng = np.random.default_rng()
        xy[:, 0] = rng.uniform(0, w, n)
        xy[:, 1] = rng.uniform(0, h, n)
        heading = rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
        speed = np.zeros(n, np.float32)
        life = np.full(n, initial_life, np.float32)
        return xy, heading, speed, life

    @staticmethod
    def spawn_near(
        parent_xy: np.ndarray, w: float, h: float, life: float
    ) -> Tuple[np.ndarray, ...]:
        """
        Spawn a single child close to `parent_xy`, wrap around toroidal edges.
        """
        rng = np.random.default_rng()
        xy = parent_xy + rng.normal(scale=10.0, size=(1, 2)).astype(np.float32)
        xy[:, 0] %= w
        xy[:, 1] %= h
        heading = rng.uniform(0.0, 2.0 * math.pi, 1).astype(np.float32)
        speed = np.zeros(1, np.float32)
        life_arr = np.full(1, life, np.float32)
        return xy, heading, speed, life_arr

    # --------------------------------------------------------------- fallback
    @staticmethod
    def random_policy(n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy blind controller – kept for backward‑compatibility.
        Returns `steer ∈ [-1,1]`, `speed_cmd ∈ [0,1]` for `n` agents.
        """
        rng = np.random.default_rng()
        steer = rng.uniform(-1.0, 1.0, n).astype(np.float32)
        speed_cmd = rng.uniform(0.0, 1.0, n).astype(np.float32)
        return steer, speed_cmd


# ----------------------------------------------------------------------------
#                         Vectorised recurrent “brain”
# ----------------------------------------------------------------------------
@dataclass
class BrainGenome:
    """Holds all trainable parameters for an *entire* population."""
    W: np.ndarray  # (Pop, n, n)
    E: np.ndarray  # (Pop, n, xdim)
    D: np.ndarray  # (Pop, ydim, n)
    b: np.ndarray  # (Pop, n)


def init_population(
    pop: int,
    neurons: int,
    input_dim: int,
    output_dim: int,
    sigma: float = 0.3,
    use_bias: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[BrainGenome, np.ndarray]:
    """
    Initialise a population of independent brains plus their hidden states.

    Returns
    -------
    genome : BrainGenome
        Batched parameter tensors.
    S : np.ndarray
        Initial hidden state, shape (Pop, neurons)
    """
    rng = rng or np.random.default_rng()
    genome = BrainGenome(
        W=rng.normal(0.0, sigma, (pop, neurons, neurons)).astype(np.float32),
        E=rng.normal(0.0, sigma, (pop, neurons, input_dim)).astype(np.float32),
        D=rng.normal(0.0, sigma, (pop, output_dim, neurons)).astype(np.float32),
        b=(np.zeros((pop, neurons), np.float32) if use_bias else 0.0),
    )
    S0 = np.zeros((pop, neurons), np.float32)
    return genome, S0


def step(
    genome: BrainGenome, S: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """
    Advance *all* agents one tick.

    Parameters
    ----------
    genome : BrainGenome
        Shared parameters batched by population index.
    S : np.ndarray
        Hidden state, shape (Pop, n). Modified IN‑PLACE for speed.
    X : np.ndarray
        Sensory input, shape (Pop, xdim)

    Returns
    -------
    Y : np.ndarray
        Output vector, shape (Pop, ydim). Caller decides how to interpret.
    """
    # S' = tanh(W·S + E·X + b)
    # einsum('pij,pj->pi') == batched GEMM
    S[:] = np.tanh(
        np.einsum("pij,pj->pi", genome.W, S)
        + np.einsum("pij,pj->pi", genome.E, X)
        + genome.b
    )

    # Y = D·S
    Y = np.einsum("pij,pj->pi", genome.D, S)
    return Y


# ---------------------------------------------------------------------------#
#                  Mutation helper for evolutionary algorithms               #
# ---------------------------------------------------------------------------#
def apply_proportional_mutation_and_decay(genome: BrainGenome, std: dict[str, float], decay_factor: float, mutation_constant: float, rng=None) -> None:
    """
    Apply proportional mutation and decay in one operation.
    Formula: parameter_value = parameter_value * (1+mutation) * decay_factor + mutation*k
    
    Parameters
    ----------
    genome : BrainGenome
        Target genome to mutate and decay.
    std : dict
        Mapping {'W':σ_w, 'E':σ_e, 'D':σ_d, 'b':σ_b} - mutation standard deviations
    decay_factor : float
        Weight decay factor (parameter *= (1 - decay_factor))
    mutation_constant : float
        Small constant for proportional mutation (k)
    """
    rng = rng or np.random.default_rng()
    decay_mult = 1.0 - decay_factor
    
    # Apply to W matrix
    if std.get("W", 0.0):
        mutation = rng.normal(0.0, std["W"], genome.W.shape).astype(np.float32)
        genome.W = genome.W * (1.0 + mutation) * decay_mult + mutation * mutation_constant
    else:
        genome.W *= decay_mult
    
    # Apply to E matrix
    if std.get("E", 0.0):
        mutation = rng.normal(0.0, std["E"], genome.E.shape).astype(np.float32)
        genome.E = genome.E * (1.0 + mutation) * decay_mult + mutation * mutation_constant
    else:
        genome.E *= decay_mult
    
    # Apply to D matrix
    if std.get("D", 0.0):
        mutation = rng.normal(0.0, std["D"], genome.D.shape).astype(np.float32)
        genome.D = genome.D * (1.0 + mutation) * decay_mult + mutation * mutation_constant
    else:
        genome.D *= decay_mult
    
    # Apply to b vector
    if isinstance(genome.b, np.ndarray):
        if std.get("b", 0.0):
            mutation = rng.normal(0.0, std["b"], genome.b.shape).astype(np.float32)
            genome.b = genome.b * (1.0 + mutation) * decay_mult + mutation * mutation_constant
        else:
            genome.b *= decay_mult


# Legacy functions kept for compatibility
def apply_weight_decay(genome: BrainGenome, decay_factor: float, rng=None) -> None:
    """Legacy function - use apply_proportional_mutation_and_decay instead"""
    apply_proportional_mutation_and_decay(genome, {}, decay_factor, 0.0, rng)


def mutate(genome: BrainGenome, std: dict[str, float], rng=None) -> None:
    """Legacy function - use apply_proportional_mutation_and_decay instead"""
    apply_proportional_mutation_and_decay(genome, std, 0.0, 0.0, rng)