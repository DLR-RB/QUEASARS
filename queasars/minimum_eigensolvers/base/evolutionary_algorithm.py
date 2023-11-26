# -*- coding: utf-8 -*-
# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit, Parameter


class Individual(ABC):
    """"""

    @abstractmethod
    def get_parameterized_quantum_circuit(
        self,
    ) -> tuple[QuantumCircuit, dict[Parameter, float]]:
        """"""
        pass


IND = TypeVar("IND", bound=Individual)


class FitnessFunction(ABC, Generic[IND]):
    """"""

    @abstractmethod
    def get_fitness(
        self, individual: IND, overwrite_angles: Optional[list[float]] = None
    ) -> "FitnessResult":
        """"""
        pass


FIT = TypeVar("FIT", bound=FitnessFunction)


@dataclass
class FitnessResult:
    """"""

    expectation_value: float
    fitness_score: float | dict[str, float]


class Population(ABC, Generic[IND]):
    """"""

    individuals: set[IND]


POP = TypeVar("POP", bound=Population)


class Operator(ABC, Generic[POP]):
    """"""

    @abstractmethod
    def apply_operator(self, population: POP) -> POP:
        """"""
        pass

    @abstractmethod
    def initialize(self, fitness_function: FIT):
        """"""
        pass
