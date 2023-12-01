# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Callable, TypeVar, Generic, Optional
from dataclasses import dataclass

from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import QuantumCircuit
from qiskit.result import QuasiDistribution

from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.minimum_eigensolvers import (
    MinimumEigensolver,
    MinimumEigensolverResult,
)

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import (
    BasePopulation,
    BasePopulationEvaluationResult,
    BaseEvolutionaryOperator,
)


POP = TypeVar("POP", bound=BasePopulation)


@dataclass
class EvolvingAnsatzMinimumEigensolverConfiguration(Generic[POP]):
    """ Configuration for the EvolvingAnsatzMinimumEigensolver

    :param population_initializer: Initialization function, which creates the initial population
        to start the evolution from
    :type population_initializer: Callable[[None], BasePopulation]
    :param evolutionary_operators: List of evolutionary operators to apply in order for each generation
    :type evolutionary_operators: list[BaseEvolutionaryOperator]
    :param estimator: Estimator primitive used for estimating the circuit's eigenvalue
    :type estimator: BaseEstimator
    :param sampler: Sampler primitive to retrieve measurement distributions with
    :type sampler: BaseSampler
    :param max_generations: Maximum number of generations to apply the evolutionary operators for
    :type max_generations: Optional[int]
    :param max_circuit_evaluations: Maximum number of circuit evaluations during the optimization
    :type max_circuit_evaluations: Optional[int]
    :param function_tolerance: Lower bound for the change in the best eigenvalue between generations
    :type function_tolerance: Optional[float]
    """

    population_initializer: Callable[[None], POP]
    evolutionary_operators: list[BaseEvolutionaryOperator[POP]]
    estimator: BaseEstimator
    sampler: BaseSampler
    max_generations: Optional[int]
    max_circuit_evaluations: Optional[int]
    step_tolerance: Optional[float]
    function_tolerance: Optional[float]
    seed: Optional[int]


class EvolvingAnsatzMinimumEigensolver(MinimumEigensolver):
    """Minimum eigensolver which uses an evolutionary algorithm to optimize the ansatz architecture in addition
    to optimizing the rotation angles of the circuit

    :param configuration: Configuration to use with this eigensolver
    :type configuration: EvolvingAnsatzMinimumEigensolverConfiguration
    """

    def __init__(
        self,
        configuration: EvolvingAnsatzMinimumEigensolverConfiguration,
    ):
        """Constructor method"""
        super().__init__()
        self._configuration = configuration

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        """
        Computes the minimum eigenvalue. The ``operator`` and ``aux_operators`` are supplied here.
        While an ``operator`` is required by algorithms, ``aux_operators`` are optional

        :param operator: Qubit operator of the observable
        :type operator: BaseOperator
        :param aux_operators: Optional list of auxiliary operators to be evaluated with the
            parameters of the minimum eigenvalue main result and their expectation values
            returned.
        :type aux_operators: ListOrDict[BaseOperator] | None

        Returns:
            An evolving ansatz minimum eigensolver result
        """
        raise NotImplementedError

    def compute_minimum_function_value(
        self,
        operator: BitstringEvaluator,
        aux_operators: ListOrDict[BitstringEvaluator] | None = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        """
        Computes the minimum function value of a function mapping bitstrings to real valued numbers.
        The ``operator`` and ``aux_operators`` are supplied here.
        While an ``operator`` is required by algorithms, ``aux_operators`` are optional

        :param operator: BitstringEvaluator which maps bitstrings to real valued numbers
        :type operator: BitstringEvaluator
        :param aux_operators: Optional list of auxiliary operators to be evaluated with the
            parameters of the minimum eigenvalue main result and their expectation values
            returned.
        :type aux_operators: ListOrDict[BaseOperator] | None

        Returns:
            An evolving ansatz minimum eigensolver result
        """
        raise NotImplementedError


class EvolvingAnsatzMinimumEigensolverResult(MinimumEigensolverResult):
    """Evolving ansatz minimum eigensolver result"""

    def __init__(self) -> None:
        super().__init__()
        self._eigenstate: QuasiDistribution | None = None
        self._optimal_parameters: dict | None = None
        self._optimal_circuit: QuantumCircuit | None = None
        self._circuit_evaluations: int | None = None
        self._generations: int | None = None
        self._final_population: BasePopulation | None = None
        self._final_population_evaluation: BasePopulationEvaluationResult | None = None

    @property
    def eigenstate(self) -> QuasiDistribution | None:
        """Return the quasi-distribution sampled from the final state

        :return: Quasi-distribution sampled from the final state
        :rtype: QuasiDistribution | None
        """
        return self._eigenstate

    @eigenstate.setter
    def eigenstate(self, value: QuasiDistribution):
        """Set the the quasi-distribution sampled from the final state

        :param value: Value to set the sampled quasi-distribution to
        :type value: QuasiDistribution
        """
        self._eigenstate = value

    @property
    def optimal_parameters(self) -> dict | None:
        """Returns the optimal parameters in a dictionary

        :return: The optimal parameters
        :rtype: dict | None
        """
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: dict) -> None:
        """Sets optimal parameters

        :param value: Value to set the optimal parameters to
        :type value: dict
        """
        self._optimal_parameters = value

    @property
    def optimal_circuit(self) -> QuantumCircuit | None:
        """The optimal circuit. Along with the optimal parameters,
        this can be used to retrieve the minimum eigenstate.

        :return: The optimal parameterized quantum circuit
        :rtype: QuantumCircuit | None
        """
        return self._optimal_circuit

    @optimal_circuit.setter
    def optimal_circuit(self, value: QuantumCircuit) -> None:
        """Sets the optimal circuit

        :param value: Value to set the optimal circuit to
        :type value: QuantumCircuit
        """
        self._optimal_circuit = value

    @property
    def circuit_evaluations(self) -> int | None:
        """Returns the number of circuit evaluations used by the eigensolver

        :return: The number of circuit evaluations
        :rtype: int
        """
        return self._circuit_evaluations

    @circuit_evaluations.setter
    def circuit_evaluations(self, value: int) -> None:
        """Sets the number of circuit evaluations used by the eigensolver

        :param value: Value to set the number of circuit evaluations to
        :type: int
        """
        self._circuit_evaluations = value

    @property
    def generations(self) -> int | None:
        """Returns the number of generations the evolutionary algorithm was run for

        :return: The number of generations the algorithm was run for
        :rtype: int | None
        """
        return self._generations

    @generations.setter
    def generations(self, value: int):
        """Sets the number of generations the evolutionary algorithm was run for

        :param value: Value to set the number of generations to
        :type value: int
        """
        self._generations = value

    @property
    def final_population(self) -> BasePopulation | None:
        """Returns the population which was evaluated during the last generation

        :return: The last population
        :rtype: BasePopulation | None
        """
        return self._final_population

    @final_population.setter
    def final_population(self, value: BasePopulation):
        """Sets the population which was evaluated during the last generation

        :param value: Value to set the last population to
        :type value: BasePopulation
        """
        self._final_population = value

    @property
    def final_population_evaluation(self):
        """Returns the evaluation results for the last population

        :return: The evaluation results for the last population
        :rtype: BasePopulationEvaluationResult | None
        """
        return self._final_population_evaluation

    @final_population_evaluation.setter
    def final_population_evaluation(self, value: BasePopulationEvaluationResult):
        """Sets the evaluation results for the last population

        :param value: Value to set the evaluation results for the last population to
        :type value: BasePopulationEvaluationResult
        """
        self._final_population_evaluation = value
