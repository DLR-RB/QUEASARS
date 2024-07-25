# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass

from qiskit.primitives.base import BaseEstimatorV2, BaseSamplerV2


@dataclass
class ConfiguredSamplerV2:
    """Dataclass that holds a qiskit SamplerV2 and the amount of shots that shall be used when sampling."""

    sampler: BaseSamplerV2
    shots: int


@dataclass
class ConfiguredEstimatorV2:
    """Dataclass that holds a qiskit EstimatorV2 and the precision to which the expectation value shall be estimated."""

    estimator: BaseEstimatorV2
    precision: float
