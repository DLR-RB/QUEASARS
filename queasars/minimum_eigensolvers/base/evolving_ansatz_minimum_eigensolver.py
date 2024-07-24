# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from typing import Callable, Generic, Optional, TypeVar, Union

from dask.distributed import Client
from numpy import median, mean

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolver

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator
from queasars.circuit_evaluation.circuit_evaluation import (
    BaseCircuitEvaluator,
    BitstringCircuitEvaluator,
    measure_quasi_distributions,
    OperatorCircuitEvaluator,
    OperatorSamplerCircuitEvaluator,
)
from queasars.circuit_evaluation.configured_primitives import ConfiguredEstimatorV2, ConfiguredSamplerV2
from queasars.circuit_evaluation.mutex_primitives import (
    BatchingMutexEstimator,
    BatchingMutexSampler,
    MutexEstimator,
    MutexSampler,
)
from queasars.circuit_evaluation.transpiling_primitives import TranspilingEstimatorV2, TranspilingSamplerV2
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import (
    BaseEvolutionaryOperator,
    BaseIndividual,
    BasePopulation,
    BasePopulationEvaluationResult,
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
    :param configured_estimator: Configured qiskit EstimatorV2 primitive used to estimate the quantum circuits'
        expectation values. If no estimator is provided, the sampler is used to calculate the expectation value instead.
    :type configured_estimator: Optional[ConfiguredEstimatorV2]
    :param configured_sampler: Configured qiskit SamplerV2 primitive used to measure the circuits QuasiDistribution.
    :type configured_sampler: BaseSampler
    :param pass_manager: A qiskit PassManager which specifies the transpilation procedure. If no pass_manager is given,
        a preset passmanager with optimization level 0 and no information on the backend is used. When running on
        real quantum hardware, the pass_manager must be user_configured to fit the backend
    :type pass_manager: PassManager
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
    :param parallel_executor: Parallel executor used for concurrent computations. Can either be a Dask Client or
        a python ThreadPool executor. If reproducible behaviour is desired only one worker thread should be used
    :type parallel_executor: Union[Client, ThreadPoolExecutor]
    :param distribution_alpha_tail: If only a Sampler is used, the expectation value is calculated from the
        probability distribution of measured basis states and their respective eigenvalues. In that case, the
        expectation value can also be calculated over only the lower alpha tail of the state distribution.
        distribution_alpha_tail can be in the range (0, 1]. By default, it is 1.
        Then the expectation is calculated over the whole state distribution. Otherwise, it is only calculated
        over the lower alpha tail of the distribution as discussed in
        https://quantum-journal.org/papers/q-2020-04-20-256/
    :type distribution_alpha_tail: float
    :param mutually_exclusive_primitives: discerns whether to only allow mutually exclusive access to the Sampler and
        Estimator primitive respectively. This is needed if the Sampler or Estimator are not threadsafe and
        a ThreadPoolExecutor with more than one thread or a Dask Client with more than one thread per process is used.
        When using a ThreadPoolExecutor with this option enabled, parallel circuit evaluations are batched.
        To accommodate non-threadsafe primitives this is enabled by default. If the sampler and estimator are threadsafe
        or the dask client is configured with only one thread per process, disabling this option may lead to performance
        improvements
    :type mutually_exclusive_primitives: bool
    """

    population_initializer: Callable[[int], POP]
    evolutionary_operators: list[BaseEvolutionaryOperator[POP]]
    configured_sampler: ConfiguredSamplerV2
    configured_estimator: Optional[ConfiguredEstimatorV2]
    pass_manager: Optional[PassManager]
    max_generations: Optional[int]
    max_circuit_evaluations: Optional[int]
    termination_criterion: Optional[EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion]
    parallel_executor: Union[Client, ThreadPoolExecutor]
    distribution_alpha_tail: float = 1
    mutually_exclusive_primitives: bool = True

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

        # If mutual exclusion is used with a ThreadpoolExecutor the circuit evaluation are batched
        if self.configuration.mutually_exclusive_primitives and isinstance(
            self.configuration.parallel_executor, ThreadPoolExecutor
        ):
            self.configuration.configured_sampler.sampler = BatchingMutexSampler(
                sampler=self.configuration.configured_sampler.sampler, waiting_duration=0.1
            )
            if self.configuration.configured_estimator is not None:
                self.configuration.configured_estimator.estimator = BatchingMutexEstimator(
                    estimator=self.configuration.configured_estimator.estimator, waiting_duration=0.1
                )

        # For mutual exclusion with a dask Client circuit evaluations are not batched
        if self.configuration.mutually_exclusive_primitives and isinstance(
            self.configuration.parallel_executor, Client
        ):
            self.configuration.configured_sampler.sampler = MutexSampler(
                sampler=self.configuration.configured_sampler.sampler
            )
            if self.configuration.configured_estimator is not None:
                self.configuration.configured_estimator.estimator = MutexEstimator(
                    estimator=self.configuration.configured_estimator.estimator
                )

        # Configure the primitives to transpile the circuit before the circuit evaluation
        if self.configuration.pass_manager is None:
            self.configuration.pass_manager = generate_preset_pass_manager(optimization_level=0)

        self.configuration.configured_sampler.sampler = TranspilingSamplerV2(
            sampler=self.configuration.configured_sampler.sampler, pass_manager=self.configuration.pass_manager
        )
        if self.configuration.configured_estimator is not None:
            self.configuration.configured_estimator.estimator = TranspilingEstimatorV2(
                estimator=self.configuration.configured_estimator.estimator,
                pass_manager=self.configuration.pass_manager,
            )

        self.logger = logging.getLogger(__name__)

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: Optional[ListOrDict[BaseOperator]] = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        """
        Computes the minimum eigenvalue. The ``operator`` and ``aux_operators`` are supplied here.
        While an ``operator`` is required by algorithms, ``aux_operators`` are optional

        :arg operator: Quantum operator of the observable. If this solver is configured to only use sampler,
                        the operator must be of type SparsePauliOp.
        :type operator: BaseOperator
        :arg aux_operators: Optional list of auxiliary operators to be evaluated with the
            parameters of the minimum eigenvalue main result and their expectation values
            returned.
        :type aux_operators: Optional[ListOrDict[BaseOperator]]

        Returns:
            An evolving ansatz minimum eigensolver result
        """
        return self.compute_minimum_eigenvalue_with_initial_state(
            operator=operator, aux_operators=aux_operators, initial_state_circuit=None
        )

    def compute_minimum_eigenvalue_with_initial_state(
        self,
        operator: BaseOperator,
        aux_operators: Optional[ListOrDict[BaseOperator]] = None,
        initial_state_circuit: Optional[QuantumCircuit] = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        """
        Computes the minimum eigenvalue. The ``operator`` and ``aux_operators`` are supplied here.
        While an ``operator`` is required by algorithms, ``aux_operators`` are optional

        :arg operator: Quantum operator of the observable. If this solver is configured to only use sampler,
                        the operator must be of type SparsePauliOp.
        :type operator: BaseOperator
        :arg aux_operators: Optional list of auxiliary operators to be evaluated with the
            parameters of the minimum eigenvalue main result and their expectation values
            returned.
        :type aux_operators: Optional[ListOrDict[BaseOperator]]
        :arg initial_state_circuit: state in which to initialize the qubits before the ansatz is applied,
            given as a quantum circuit. The quantum circuit must operate on exactly as many qubits
            as the given operator
        :type initial_state_circuit: Optional[QuantumCircuit]

        Returns:
            An evolving ansatz minimum eigensolver result
        """

        def estimator_initialisation_function(op: BaseOperator) -> OperatorCircuitEvaluator:
            if self.configuration.configured_estimator is None:
                raise ValueError("This error should never occur! This seems to be an issue of the internal logic!")
            return OperatorCircuitEvaluator(
                estimator=self.configuration.configured_estimator.estimator,
                estimator_precision=self.configuration.configured_estimator.precision,
                operator=op,
                initial_state_circuit=initial_state_circuit,
            )

        def sampler_initialisation_function(op: BaseOperator) -> OperatorSamplerCircuitEvaluator:
            if self.configuration.configured_sampler is None:
                raise ValueError("This error should never occur! This seems to be an issue of the internal logic!")
            if not isinstance(op, SparsePauliOp):
                raise ValueError(
                    "The operator must be of type SparsePauliOp, when using a Sampler "
                    + f"to approximate the expectation value! Instead it was of type {type(op)}!"
                )
            return OperatorSamplerCircuitEvaluator(
                sampler=self.configuration.configured_sampler.sampler,
                sampler_shots=self.configuration.configured_sampler.shots,
                operator=op,
                alpha=self.configuration.distribution_alpha_tail,
                initial_state_circuit=initial_state_circuit,
            )

        evaluator_initialisation_function: Callable[[BaseOperator], BaseCircuitEvaluator]
        if self.configuration.configured_estimator is not None:
            evaluator_initialisation_function = estimator_initialisation_function
        else:
            evaluator_initialisation_function = sampler_initialisation_function

        evaluator: BaseCircuitEvaluator
        evaluator = evaluator_initialisation_function(operator)

        aux_evaluators: Optional[ListOrDict[BaseCircuitEvaluator]] = None
        if aux_operators is None:
            aux_evaluators = None
        if isinstance(aux_operators, list):
            aux_evaluators = [evaluator_initialisation_function(aux_operator) for aux_operator in aux_operators]
        if isinstance(aux_operators, dict):
            aux_evaluators = {
                key: evaluator_initialisation_function(aux_operator) for key, aux_operator in aux_operators.items()
            }

        return self._solve_by_evolution(
            circuit_evaluator=evaluator,
            aux_circuit_evaluators=aux_evaluators,
            initial_state_circuit=initial_state_circuit,
        )

    def compute_minimum_function_value(
        self,
        operator: BitstringEvaluator,
        aux_operators: Optional[ListOrDict[BitstringEvaluator]] = None,
        initial_state_circuit: Optional[QuantumCircuit] = None,
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
        :arg initial_state_circuit: state in which to initialize the qubits before the ansatz is applied,
            given as a quantum circuit. The quantum circuit must operate on exactly as many qubits
            as the input length specified by the BitstringEvaluator
        :type initial_state_circuit:  Optional[QuantumCircuit]

        Returns:
            An evolving ansatz minimum eigensolver result
        """

        def evaluator_initialisation_function(op: BitstringEvaluator):
            return BitstringCircuitEvaluator(
                sampler=self.configuration.configured_sampler.sampler,
                sampler_shots=self.configuration.configured_sampler.shots,
                bitstring_evaluator=op,
                alpha=self.configuration.distribution_alpha_tail,
                initial_state_circuit=initial_state_circuit,
            )

        evaluator: BaseCircuitEvaluator = evaluator_initialisation_function(op=operator)

        aux_evaluators: ListOrDict[BaseCircuitEvaluator] = []
        if aux_operators is None:
            aux_evaluators = []
        if isinstance(aux_operators, list):
            aux_evaluators = [evaluator_initialisation_function(aux_operator) for aux_operator in aux_operators]
        if isinstance(aux_operators, dict):
            aux_evaluators = {
                key: evaluator_initialisation_function(aux_operator) for key, aux_operator in aux_operators.items()
            }

        return self._solve_by_evolution(
            circuit_evaluator=evaluator,
            aux_circuit_evaluators=aux_evaluators,
            initial_state_circuit=initial_state_circuit,
        )

    def _solve_by_evolution(
        self,
        circuit_evaluator: BaseCircuitEvaluator,
        aux_circuit_evaluators: ListOrDict[BaseCircuitEvaluator],
        initial_state_circuit: Optional[QuantumCircuit] = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        n_circuit_evaluations: list[int] = []
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
            self.logger.info("Current best expectation value: %f" % evaluation_result.best_expectation_value)
            filtered_expectations = [
                expectation for expectation in evaluation_result.expectation_values if expectation is not None
            ]
            self.logger.info("Current median expectation value: %f" % median(filtered_expectations))
            self.logger.info("Current average expectation value: %f" % mean(filtered_expectations))

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
            nonlocal n_generations
            if len(n_circuit_evaluations) < n_generations + 1:
                n_circuit_evaluations.append(evaluations)
            else:
                n_circuit_evaluations[n_generations] += evaluations

        operator_context: OperatorContext = OperatorContext(
            circuit_evaluator=circuit_evaluator,
            result_callback=result_callback,
            circuit_evaluation_count_callback=circuit_evaluation_callback,
            parallel_executor=self.configuration.parallel_executor,
        )

        population: BasePopulation = self.configuration.population_initializer(circuit_evaluator.n_qubits)

        self.logger.info("Starting evolution!")

        while not terminate:

            for operator in self.configuration.evolutionary_operators:

                # Terminate if the maximum circuit evaluation count has been exceeded
                if (
                    self.configuration.max_circuit_evaluations is not None
                    and sum(n_circuit_evaluations) >= self.configuration.max_circuit_evaluations
                ):
                    terminate = True

                # Terminate if the next operation is expected to exceed the circuit evaluation count
                estimated_evaluations: Optional[int] = operator.get_n_expected_circuit_evaluations(
                    population=population, operator_context=operator_context
                )
                if (
                    self.configuration.max_circuit_evaluations is not None
                    and estimated_evaluations is not None
                    and sum(n_circuit_evaluations) + estimated_evaluations >= self.configuration.max_circuit_evaluations
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

        best_circuit: QuantumCircuit = current_best_individual.get_parameterized_quantum_circuit()
        if initial_state_circuit is not None:
            best_circuit = initial_state_circuit.compose(best_circuit, inplace=False)
        best_circuit.measure_all(inplace=True)

        result = EvolvingAnsatzMinimumEigensolverResult()
        result.eigenvalue = current_best_expectation_value
        result.eigenstate = measure_quasi_distributions(
            circuits=[best_circuit],
            parameter_values=[list(current_best_individual.get_parameter_values())],
            sampler=self.configuration.configured_sampler.sampler,
            shots=self.configuration.configured_sampler.shots,
        )[0]
        result.best_individual = current_best_individual
        result.circuit_evaluations = n_circuit_evaluations
        result.generations = n_generations
        result.population_evaluation_results = population_evaluations
        result.initial_state_circuit = initial_state_circuit

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
