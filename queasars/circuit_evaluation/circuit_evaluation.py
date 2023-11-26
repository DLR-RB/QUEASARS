# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator


class CircuitEvaluator(ABC):
    """"""

    @abstractmethod
    def evaluate_circuit(self, circuit: QuantumCircuit, angles: list[float]) -> float:
        """"""
        pass


class ObservableCircuitEvaluator(CircuitEvaluator):
    """"""

    def __init__(self, estimator: BaseEstimator):
        """"""
        pass

    def evaluate_circuit(self, circuit: QuantumCircuit, angles: list[float]) -> float:
        pass


class BitstringCircuitEvaluator(CircuitEvaluator):
    def __init__(self, sampler: BaseSampler, bitstring_evaluator: BitstringEvaluator):
        """"""
        pass

    def evaluate_circuit(self, circuit: QuantumCircuit, angles: list[float]) -> float:
        pass
