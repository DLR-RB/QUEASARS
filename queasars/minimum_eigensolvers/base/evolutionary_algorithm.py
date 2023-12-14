# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Callable
from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from dask.distributed import Client

from queasars.circuit_evaluation.circuit_evaluation import BaseCircuitEvaluator


class BaseIndividual(ABC):
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
    def get_parameter_values(self) -> tuple[float, ...]:
        """
        Get the parameter values for the parameterized quantum circuit, as specified in this individual's genome

        :return: A list of rotation angles
        :rtype: tuple[float, ...]
        """

    @abstractmethod
    def __hash__(self):
        pass


IND = TypeVar("IND", bound=BaseIndividual)


@dataclass
class BasePopulation(ABC, Generic[IND]):
    """Base class representing the state of a population of individuals in an evolutionary algorithm

    :param individuals: Tuple of all individuals in this population
    :type individuals: tuple[BaseIndividual, ...]
    """

    individuals: tuple[IND, ...]


POP = TypeVar("POP", bound=BasePopulation)


@dataclass
class BasePopulationEvaluationResult(ABC, Generic[IND]):
    """Base class representing the result of evaluating a population in an evolutionary algorithm

    :param population: Population which was evaluated
    :type population: BasePopulation
    :param expectation_values: Dictionary containing expectation values for the individuals of the evaluated population
    :type expectation_values: dict[BaseIndividual, Optional[float]]
    :param best_individual: Best individual from within the evaluated population
    :type best_individual: BaseIndividual
    """

    population: BasePopulation[IND]
    expectation_values: dict[IND, Optional[float]]
    best_individual: IND


@dataclass
class OperatorContext:
    """Dataclass containing additional references needed by Operators

    :param circuit_evaluator: CircuitEvaluator used to get the expectation value of the circuits (Individuals)
    :type circuit_evaluator: BaseCircuitEvaluator
    :param result_callback: Callback function to report results from evaluating a population
    :type result_callback: Callable[[BasePopulationEvaluationResult], None]
    :param circuit_evaluation_count_callback: Callback functions to report the number of circuit evaluations
        used by an operator.
    :type circuit_evaluation_count_callback: Callable[[Int], None]
    :param dask_client: Dask client to use for task parallelization
    :type: Client

    """

    circuit_evaluator: BaseCircuitEvaluator
    result_callback: Callable[[BasePopulationEvaluationResult], None]
    circuit_evaluation_count_callback: Callable[[int], None]
    dask_client: Client


class BaseEvolutionaryOperator(ABC, Generic[POP]):
    """Base class representing any evolutionary operator, which maps from a population to a new population"""

    @abstractmethod
    def apply_operator(
        self,
        population: POP,
        operator_context: OperatorContext,
    ) -> POP:
        """
        Applies the operator to the population and returns a new, changed population. The original population
        remains unchanged

        :arg population: Population to apply the operator to
        :type population: BasePopulation
        :arg operator_context:
        :type operator_context:
        """

    @abstractmethod
    def get_n_expected_circuit_evaluations(self, population: POP) -> Optional[int]:
        """Returns the expected amount of circuit evaluations needed for applying
        this operator to the given population. Since this is often probabilistic,
        this is only an estimate.

        :arg population: for which the amount of expected circuit evaluations shall be estimated
        :type population: BasePopulation
        :return: the amount of expected circuit evaluations needed or None if no estimate can be given
        :rtype: Optional[int]
        """
