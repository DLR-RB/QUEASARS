# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit

from queasars.circuit_evaluation.circuit_evaluation import CircuitEvaluator


class Individual(ABC):
    """Base class representing an individual genome in an evolutionary algorithm. For the purposes of evolving
    ansatz eigensolvers such an individual genome represents a parameterized quantum circuit with a defined list
    of rotation angles"""

    @abstractmethod
    def get_quantum_circuit(self) -> QuantumCircuit:
        """
        Builds the quantum circuit as specified by this genome using the appropriate rotation angles.

        :return: A non parameterized quantum circuit
        :rtype: QuantumCircuit
        """

    @abstractmethod
    def get_parameterized_quantum_circuit(
        self,
    ) -> QuantumCircuit:
        """
        Builds a parameterized quantum circuit as specified in this individual's genome. The returned circuit is only
        an appropriate representation of this individual, if the parameters are manually populated by the angles
        as returned by `get_rotation_angles()`

        :return: A parameterized quantum circuit
        :rtype: QuantumCircuit
        """

    @abstractmethod
    def get_rotation_angles(self) -> list[float]:
        """
        Get the rotation angles for the parameterized quantum circuit, as specified in this individual's genome

        :return: A list of rotation angles
        :rtype: list[float]
        """

    @abstractmethod
    def __hash__(self):
        pass


IND = TypeVar("IND", bound=Individual)


@dataclass
class Population(ABC, Generic[IND]):
    """Base class representing the state of a population of individuals in an evolutionary algorithm

    :param individuals: Set of all individuals in this population
    :type individuals: set[Individual]
    :param circuit_evaluations: All currently gathered circuit evaluation values.
    :type circuit_evaluations: dict[Individual, Optional[float]]
    """

    individuals: set[IND]
    circuit_evaluations: dict[IND, Optional[float]]


POP = TypeVar("POP", bound=Population)


class Operator(ABC, Generic[POP]):
    """Base class representing any evolutionary operator, which maps from a population to a new population"""

    @abstractmethod
    def apply_operator(
        self,
        population: POP,
        circuit_evaluator: CircuitEvaluator,
        best_individual_callback: Optional[Callable[[Individual, float], None]] = None,
    ) -> POP:
        """
        Applies the operator to the population and returns a new, changed population. The original population
        remains unchanged

        :param population: Population to apply the operator to
        :type population: Population
        :param circuit_evaluator: Circuit evaluator to evaluate individuals with
        :type circuit_evaluator: CircuitEvaluator
        :param best_individual_callback: Callback to report good individuals and their circuit evaluation results
        :type best_individual_callback: Optional[Callable[[Individual, Float], None]]
        :return: The new population changed by the operator
        :rtype: Population
        """


class EvolutionaryAlgorithm(Generic[POP]):
    """Implementation of an evolutionary algorithm using the primitives defined in this module.

    :param initial_population: Population to start the evolution from.
    :type initial_population: Population
    :param evolutionary_operators: Operators to apply in sequence to the population for each generation.
    :type evolutionary_operators: list[Operator]
    :param n_generations: Number of generations during which the operators are applied.
    :type n_generations: int
    :param circuit_evaluator: Circuit evaluator to evaluate individuals with
    :type circuit_evaluator: CircuitEvaluator
    :param termination_callback: Optional callback used to check after each generation whether the evolution can be
        stopped prematurely. Return true to terminate the evolution. Arguments are: number of current generation,
        amount of called circuit evaluations, best circuit evaluation value so far
    :type termination_callback: Optional[Callable[[int, int, float], bool]]
    """

    def __init__(
        self,
        initial_population: POP,
        evolutionary_operators: list[Operator],
        n_generations: int,
        circuit_evaluator: CircuitEvaluator,
        termination_callback: Optional[Callable[[int, int, float], bool]] = None,
    ):
        """Constructor method"""

    def optimize(self) -> "EvolutionaryAlgorithmResult":
        """Runs the evolutionary optimization until `n_generations` is exhausted or the `termination_callback`
        returns true.

        :return: An `EvolutionaryAlgorithmResult`
        :rtype: EvolutionaryAlgorithm
        """
        raise NotImplementedError


@dataclass
class EvolutionaryAlgorithmResult:
    """Dataclass containing the results from running an evolutionary algorithm

    :param final_population: State of the population after the last generation
    :type final_population: Population
    :param best_individual: Individual with the best circuit evaluation found during all generations
    :type best_individual: Individual
    :param best_circuit_evaluation: Circuit evaluation value of the `best_individual`
    """

    final_population: Population
    best_individual: Individual
    best_circuit_evaluation: float
