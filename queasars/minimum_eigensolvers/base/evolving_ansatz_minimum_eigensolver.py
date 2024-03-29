# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from dataclasses import dataclass
import logging
from typing import Callable, TypeVar, Generic, Optional, Union

from dask.distributed import Client, LocalCluster
from qiskit.primitives import BaseEstimator, BaseSampler, SamplerResult
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolver

from queasars.circuit_evaluation.circuit_evaluation import (
    BaseCircuitEvaluator,
    OperatorCircuitEvaluator,
    BitstringCircuitEvaluator,
)
from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import (
    BaseIndividual,
    BasePopulation,
    BasePopulationEvaluationResult,
    BaseEvolutionaryOperator,
    OperatorContext,
)
from queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver_result import (
    EvolvingAnsatzMinimumEigensolverResult,
)
from queasars.minimum_eigensolvers.base.termination_criteria import (
    EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion,
)


POP = TypeVar("POP", bound=BasePopulation)


@dataclass
class EvolvingAnsatzMinimumEigensolverConfiguration(Generic[POP]):
    """Configuration for the EvolvingAnsatzMinimumEigensolver

    :param population_initializer: Initialization function, which creates the initial population
        to start the evolution from, for a given problem size (in qubit needed)
    :type population_initializer: Callable[[int], BasePopulation]
    :param evolutionary_operators: List of evolutionary operators to apply in order for each generation
    :type evolutionary_operators: list[BaseEvolutionaryOperator]
    :param estimator: Estimator primitive used to estimate the circuit's eigenvalue. If none is provided for that
        purpose, the sampler is used instead. If reproducible behaviour is required, the seed option of the estimator
        needs to be set
    :type estimator: Optional[BaseEstimator]
    :param sampler: Sampler primitive used to measure the circuits QuasiDistribution. If reproducible behaviour is
        required, the seed option of the sampler needs to be set
    :type sampler: BaseSampler
    :param max_generations: Maximum amount of generations the evolution may go on for. Either max_generations or
        max_circuit_evaluations or termination_criterion needs to be provided
    :type max_generations: Optional[int]
    :param max_circuit_evaluations: Maximum amount of circuit evaluations the solver may use. Depending
        on the configuration this measure may be undershot or overshot significantly. Either max_generations or
        max_circuit_evaluations or termination_criterion needs to be provided
    :type max_circuit_evaluations: Optional[int]
    :param termination_criterion: criterion which defines how to determine whether the solver has converged.
        Either max_generations or max_circuit_evaluations or termination_criterion needs to be provided
    :type termination_criterion: Optional[EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion]
    :param dask_client: optional dask client to facilitate parallelization, if None is given, a dask local cluster using
        multiprocessing is spun up for that purpose
    :type dask_client: Optional[Client]
    """

    population_initializer: Callable[[int], POP]
    evolutionary_operators: list[BaseEvolutionaryOperator[POP]]
    estimator: Optional[BaseEstimator]
    sampler: BaseSampler
    max_generations: Optional[int]
    max_circuit_evaluations: Optional[int]
    termination_criterion: Optional[EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion]
    dask_client: Optional[Client]

    def __post_init__(self):
        if self.max_generations is None and self.max_circuit_evaluations is None and self.termination_criterion is None:
            raise ValueError(
                "At least one of the parameters max_generations, max_circuit_evaluations or"
                + "termination_criterion must not be None!"
            )


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
        self.configuration = configuration
        self.logger = logging.getLogger(__name__)

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: Optional[ListOrDict[BaseOperator]] = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        """
        Computes the minimum eigenvalue. The ``operator`` and ``aux_operators`` are supplied here.
        While an ``operator`` is required by algorithms, ``aux_operators`` are optional

        :arg operator: Qubit operator of the observable
        :type operator: BaseOperator
        :arg aux_operators: Optional list of auxiliary operators to be evaluated with the
            parameters of the minimum eigenvalue main result and their expectation values
            returned.
        :type aux_operators: Optional[ListOrDict[BaseOperator]]

        Returns:
            An evolving ansatz minimum eigensolver result
        """
        evaluation_primitive: Union[BaseEstimator, BaseSampler]
        if self.configuration.estimator is not None:
            evaluation_primitive = self.configuration.estimator
        else:
            evaluation_primitive = self.configuration.sampler

        evaluator: BaseCircuitEvaluator
        evaluator = OperatorCircuitEvaluator(qiskit_primitive=evaluation_primitive, operator=operator)

        aux_evaluators: Optional[ListOrDict[BaseCircuitEvaluator]]
        if aux_operators is None:
            aux_evaluators = None
        if isinstance(aux_operators, list):
            aux_evaluators = [
                OperatorCircuitEvaluator(qiskit_primitive=evaluation_primitive, operator=aux_operator)
                for aux_operator in aux_operators
            ]
        if isinstance(aux_operators, dict):
            aux_evaluators = {
                key: OperatorCircuitEvaluator(qiskit_primitive=evaluation_primitive, operator=aux_operator)
                for key, aux_operator in aux_operators.items()
            }

        return self._solve_by_evolution(circuit_evaluator=evaluator, aux_circuit_evaluators=aux_evaluators)

    def compute_minimum_function_value(
        self,
        operator: BitstringEvaluator,
        aux_operators: Optional[ListOrDict[BitstringEvaluator]] = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        """
        Computes the minimum function value of a function mapping bitstrings to real valued numbers.
        The ``operator`` and ``aux_operators`` are supplied here.
        While an ``operator`` is required by algorithms, ``aux_operators`` are optional

        :arg operator: BitstringEvaluator which maps bitstrings to real valued numbers
        :type operator: BitstringEvaluator
        :arg aux_operators: Optional list of auxiliary operators to be evaluated with the
            parameters of the minimum eigenvalue main result and their expectation values
            returned.
        :type aux_operators: Optional[ListOrDict[BaseOperator]]

        Returns:
            An evolving ansatz minimum eigensolver result
        """
        evaluator: BaseCircuitEvaluator = BitstringCircuitEvaluator(
            sampler=self.configuration.sampler, bitstring_evaluator=operator
        )

        aux_evaluators: ListOrDict[BaseCircuitEvaluator]
        if aux_operators is None:
            aux_evaluators = []
        if isinstance(aux_operators, list):
            aux_evaluators = [
                BitstringCircuitEvaluator(sampler=self.configuration.sampler, bitstring_evaluator=aux_operator)
                for aux_operator in aux_operators
            ]
        if isinstance(aux_operators, dict):
            aux_evaluators = {
                key: BitstringCircuitEvaluator(sampler=self.configuration.sampler, bitstring_evaluator=aux_operator)
                for key, aux_operator in aux_operators.items()
            }

        return self._solve_by_evolution(circuit_evaluator=evaluator, aux_circuit_evaluators=aux_evaluators)

    def _solve_by_evolution(
        self,
        circuit_evaluator: BaseCircuitEvaluator,
        aux_circuit_evaluators: ListOrDict[BaseCircuitEvaluator],
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        n_circuit_evaluations: int = 0
        n_generations: int = 0
        terminate: bool = False
        current_best_individual: Optional[BaseIndividual] = None
        current_best_expectation_value: Optional[float] = None
        population_evaluations: list[BasePopulationEvaluationResult] = []
        if self.configuration.termination_criterion is not None:
            self.configuration.termination_criterion.reset_state()

        def result_callback(evaluation_result: BasePopulationEvaluationResult) -> None:
            nonlocal current_best_individual
            nonlocal current_best_expectation_value
            nonlocal population_evaluations
            nonlocal terminate
            nonlocal n_generations

            population_evaluations.append(evaluation_result)

            if (
                current_best_individual is None
                or current_best_expectation_value is None
                or evaluation_result.best_expectation_value < current_best_expectation_value
            ):
                current_best_individual = evaluation_result.best_individual
                current_best_expectation_value = evaluation_result.best_expectation_value

            self.logger.info(f"Results for generation: {n_generations}")
            self.logger.info("Expectation value of best individual found so far: %d" % current_best_expectation_value)
            filtered_expectations = [
                expectation for expectation in evaluation_result.expectation_values if expectation is not None
            ]
            self.logger.info(
                "Average expectation value in the population currently: %d"
                % (sum(filtered_expectations) / len(filtered_expectations))
            )

            n_generations += 1

            if self.configuration.termination_criterion is not None:
                if current_best_individual is None or current_best_expectation_value is None:
                    raise Exception("No current best individual was determined before calling the termination check!")
                terminate = self.configuration.termination_criterion.check_termination(
                    population_evaluation=evaluation_result,
                    best_individual=current_best_individual,
                    best_expectation_value=current_best_expectation_value,
                )

        def circuit_evaluation_callback(evaluations: int) -> None:
            nonlocal n_circuit_evaluations
            n_circuit_evaluations += evaluations

        client: Client
        if self.configuration.dask_client is not None:
            client = self.configuration.dask_client
        else:
            cluster = LocalCluster(processes=True)
            client = cluster.get_client()
            self.configuration.dask_client = client

        operator_context: OperatorContext = OperatorContext(
            circuit_evaluator=circuit_evaluator,
            result_callback=result_callback,
            circuit_evaluation_count_callback=circuit_evaluation_callback,
            dask_client=client,
        )

        population: BasePopulation = self.configuration.population_initializer(circuit_evaluator.n_qubits)

        self.logger.info("Starting evolution!")

        while not terminate:

            for operator in self.configuration.evolutionary_operators:

                # Terminate if the maximum circuit evaluation count has been exceeded
                if (
                    self.configuration.max_circuit_evaluations is not None
                    and n_circuit_evaluations >= self.configuration.max_circuit_evaluations
                ):
                    terminate = True

                # Terminate if the next operation is expected to exceed the circuit evaluation count
                estimated_evaluations: Optional[int] = operator.get_n_expected_circuit_evaluations(
                    population=population, operator_context=operator_context
                )
                if (
                    self.configuration.max_circuit_evaluations is not None
                    and estimated_evaluations is not None
                    and n_circuit_evaluations + estimated_evaluations >= self.configuration.max_circuit_evaluations
                ):
                    terminate = True

                # Terminate if the maximum amount of generations have passed
                if (
                    self.configuration.max_generations is not None
                    and n_generations >= self.configuration.max_generations
                ):
                    terminate = True

                if terminate:
                    break

                population = operator.apply_operator(population=population, operator_context=operator_context)

        if (
            current_best_individual is None
            or current_best_expectation_value is None
            or len(population_evaluations) == 0
        ):
            raise Exception("The algorithm seems to have terminated without having evaluated any population! ")

        sampler_result_best_individual: SamplerResult = self.configuration.sampler.run(
            current_best_individual.get_quantum_circuit().measure_all(inplace=False)
        ).result()

        result = EvolvingAnsatzMinimumEigensolverResult()
        result.eigenvalue = current_best_expectation_value
        result.eigenstate = sampler_result_best_individual.quasi_dists[0]
        result.optimal_circuit = current_best_individual.get_parameterized_quantum_circuit()
        result.optimal_parameters = dict(
            zip(result.optimal_circuit.parameters, current_best_individual.get_parameter_values())
        )
        result.circuit_evaluations = n_circuit_evaluations
        result.generations = n_generations
        result.population_evaluation_results = population_evaluations

        if isinstance(aux_circuit_evaluators, list):
            result.aux_operators_evaluated = [
                evaluator.evaluate_circuits(
                    [current_best_individual.get_parameterized_quantum_circuit()],
                    [list(current_best_individual.get_parameter_values())],
                )[0]
                for evaluator in aux_circuit_evaluators
            ]
        if isinstance(aux_circuit_evaluators, dict):
            result.aux_operators_evaluated = {
                name: evaluator.evaluate_circuits(
                    [current_best_individual.get_parameterized_quantum_circuit()],
                    [list(current_best_individual.get_parameter_values())],
                )[0]
                for name, evaluator in aux_circuit_evaluators.items()
            }

        return result
