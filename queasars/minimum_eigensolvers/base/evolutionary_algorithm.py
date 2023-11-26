# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit

from queasars.circuit_evaluation.circuit_evaluation import CircuitEvaluator


class Individual(ABC):
    """"""

    @abstractmethod
    def get_parameterized_quantum_circuit(
        self,
    ) -> QuantumCircuit:
        """"""
        pass

    @abstractmethod
    def get_rotation_angles(self) -> list[float]:
        """"""
        pass


IND = TypeVar("IND", bound=Individual)


@dataclass
class Population(ABC, Generic[IND]):
    """"""

    individuals: set[IND]
    evaluated_individuals: dict[IND, float]


POP = TypeVar("POP", bound=Population)


class Operator(ABC, Generic[POP]):
    """"""

    @abstractmethod
    def apply_operator(self, population: POP, circuit_evaluator: CircuitEvaluator) -> POP:
        """"""
        pass


class EvolutionaryAlgorithm(Generic[POP]):
    """"""

    def __init__(
        self,
        initial_population: POP,
        evolutionary_operators: list[Operator],
        n_generations: int,
        circuit_evaluator: CircuitEvaluator,
        callback: Optional[Callable[[int, int, float], bool]] = None,
    ):
        """"""
        pass

    def optimize(self) -> "EvolutionaryAlgorithmResult":
        """"""
        pass


@dataclass
class EvolutionaryAlgorithmResult:
    final_population: Population
    best_individual: Individual
    best_circuit_evaluation: float
