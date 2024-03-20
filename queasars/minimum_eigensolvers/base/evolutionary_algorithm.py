# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Callable, Union

from dask.distributed import Client
from qiskit.circuit import QuantumCircuit

from queasars.circuit_evaluation.circuit_evaluation import BaseCircuitEvaluator


class BaseIndividual(ABC):
    """Base class representing an individual genome in an evolutionary algorithm. For the purposes of evolving
    ansatz eigensolvers such an individual genome represents a parameterized quantum circuit with a defined list
    of rotation angles"""

    def get_quantum_circuit(self) -> QuantumCircuit:
        """
        Builds the quantum circuit as specified by this genome using the appropriate parameter values

        :return: A non parameterized quantum circuit
        :rtype: QuantumCircuit
        """
        return self.get_parameterized_quantum_circuit().assign_parameters(parameters=self.get_parameter_values())

    @abstractmethod
    def get_parameterized_quantum_circuit(
        self,
    ) -> QuantumCircuit:
        """
        Builds a parameterized quantum circuit as specified in this individual's genome. The returned circuit is only
        an appropriate representation of this individual, if the parameters are manually populated by the angles
        as returned by `get_parameter_values()`

        :return: A parameterized quantum circuit
        :rtype: QuantumCircuit
        """

    @abstractmethod
    def get_parameter_values(self) -> tuple[float, ...]:
        """
        Get the parameter values for the parameterized quantum circuit, as specified in this individual's genome

        :return: A tuple of parameter values
        :rtype: tuple[float, ...]
        """

    @abstractmethod
    def __eq__(self, other):
        pass

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
    :param expectation_values: Expectation values for the individuals of the evaluated population.
        The indices of the expectation values match the indices of the individuals in population.individuals
    :type expectation_values: tuple[Optional[float], ...]
    :param best_individual: Best individual from within the evaluated population
    :type best_individual: BaseIndividual
    :param best_expectation_value: Expectation value for the best individual
    :type best_expectation_value: float
    """

    population: BasePopulation[IND]
    expectation_values: tuple[Optional[float], ...]
    best_individual: IND
    best_expectation_value: float


@dataclass
class OperatorContext:
    """Dataclass containing additional references needed by Operators

    :param circuit_evaluator: CircuitEvaluator used to get the expectation value of the circuits (Individuals)
    :type circuit_evaluator: BaseCircuitEvaluator
    :param result_callback: Callback function to report results from evaluating a population. Calling this
        Callback marks the end of the current generation after the current operation has finished
    :type result_callback: Callable[[BasePopulationEvaluationResult], None]
    :param circuit_evaluation_count_callback: Callback functions to report the number of circuit evaluations
        used by an operator.
    :type circuit_evaluation_count_callback: Callable[[Int], None]
    :param parallel_executor: Parallel executor used for concurrent computations. Can either be a Dask Client or
        a python ThreadPool executor
    :type parallel_executor: Union[Client, ThreadPoolExecutor]
    """

    circuit_evaluator: BaseCircuitEvaluator
    result_callback: Callable[[BasePopulationEvaluationResult], None]
    circuit_evaluation_count_callback: Callable[[int], None]
    parallel_executor: Union[Client, ThreadPoolExecutor]


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
        :arg operator_context: additional context needed by the operator
        :type operator_context: OperatorContext
        """

    @abstractmethod
    def get_n_expected_circuit_evaluations(self, population: POP, operator_context: OperatorContext) -> Optional[int]:
        """Returns the expected amount of circuit evaluations needed for applying
        this operator to the given population. Since this is often probabilistic,
        this is only an estimate.

        :arg population: for which the amount of expected circuit evaluations shall be estimated
        :type population: BasePopulation
        :arg operator_context: additional context needed by the Operator
        :type operator_context: OperatorContext
        :return: the amount of expected circuit evaluations needed or None if no estimate can be given
        :rtype: Optional[int]
        """
