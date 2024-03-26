# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from random import Random
from typing import Callable, Optional, Union

from dask.distributed import Client

from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.optimizers import Optimizer

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BaseEvolutionaryOperator
from queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver import (
    EvolvingAnsatzMinimumEigensolver,
    EvolvingAnsatzMinimumEigensolverResult,
    EvolvingAnsatzMinimumEigensolverConfiguration,
)
from queasars.minimum_eigensolvers.base.termination_criteria import (
    EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.population import EVQEPopulation
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.speciation import EVQESpeciation
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.selection import EVQESelection
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.mutation import (
    EVQELastLayerParameterSearch,
    EVQEParameterSearch,
    EVQETopologicalSearch,
    EVQELayerRemoval,
)
from queasars.utility.random import new_random_seed


@dataclass
class EVQEMinimumEigensolverConfiguration:
    """Configuration for the EVQEMinimumEigensolver

    :param estimator: Estimator primitive used to estimate the circuit's eigenvalue. If none is provided for that
        purpose, the sampler is used instead. If reproducible behaviour is required, the seed option of the estimator
        needs to be set. If a dask Client is used as the parallel_executor,
        the Estimator needs to be serializable by dask, otherwise the computation will fail
    :type estimator: Optional[Estimator]
    :param sampler: Sampler primitive used to measure the circuits QuasiDistribution. If reproducible behaviour is
        required, the seed option of the estimator needs to be set. If a dask Client is used as the parallel_executor,
        the Sampler needs to be serializable by dask, otherwise the computation will fail
    :type sampler: Sampler
    :param optimizer: Qiskit optimizer used to optimize the parameter values. Should be configured to terminate after
        a relatively low amount of circuit evaluations to enable the gradual evolution of the individuals
    :type optimizer: Optimizer
    :param optimizer_n_circuit_evaluations: amount of circuit evaluations expected per optimizer run, None if unknown
    :type optimizer_n_circuit_evaluations: Optional[int]
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
    :param random_seed: Optional integer value to control randomness. For this to be effective the seed option for
        the sampler (and if applicable the estimator) need to be set and the dask client must use one separate process
        per worker
    :type random_seed: Optional[int]
    :param population_size: amount of individuals within a population
    :type population_size: int
    :param randomize_initial_population_parameters: Determines whether the parameter values of the individuals in
        the first population shall be initialized randomly or at 0. If during the optimization process no improvement
        occurs at all this may be due to the optimizer not being able to detect a gradient at parameter values of 0
    :type randomize_initial_population_parameters: bool
    :param speciation_genetic_distance_threshold: Genetic distance (amount of circuit layers not shared
        between two individuals) at which an individual belongs to a new species
    :type speciation_genetic_distance_threshold: int
    :param selection_alpha_penalty: Penalty added to the individuals fitness for each circuit layer. This can
        be thought of as the minimum gain in expectation value needed for an additional circuit layer to pay off
    :type selection_alpha_penalty: float
    :param selection_beta_penalty: Penalty added to the individuals fitness for each controlled rotation gate
        in the quantum circuit represented by the individual. This can be thought of as the minimum gain in
        expectation value needed for an additional controlled rotation gate to pay off
    :type selection_beta_penalty: float
    :param parameter_search_probability: probability with which the parameter search mutation is applied to an
        individual during one generation. Must be in the range [0, 1]
    :type parameter_search_probability: float
    :param topological_search_probability: probability with which the topological search mutation is applied to an
        individual during one generation. Must be in the range [0, 1]
    :type topological_search_probability: float
    :param layer_removal_probability: probability with which the layer removal mutation is applied to an individual
        during one generation. Must be in the range [0, 1]
    :param parallel_executor: Parallel executor used for concurrent computations. Can either be a Dask Client or
        a python ThreadPool executor. If a dask Client is used, both the Sampler and Estimator need to be serializable
        by dask, otherwise the computation will fail. If no parallel_executor is provided a ThreadPoolExecutor
        with as many threads as population_size will be launched
    :type parallel_executor: Union[Client, ThreadPoolExecutor, None]
    :param mutually_exclusive_primitives: discerns whether to only allow mutually exclusive access to the Sampler and
        Estimator primitive respectively. This is needed if the Sampler or Estimator are not threadsafe and
        a ThreadPoolExecutor with more than one thread or a Dask Client with more than one thread per process is used.
        When using a ThreadPoolExecutor with this option enabled, parallel circuit evaluations are batched.
        For safety reasons this is enabled by default. If the sampler and estimator are threadsafe or the dask client
        is configured with only one thread per process, disabling this option may lead to performance improvements
    :type mutually_exclusive_primitives: bool
    """

    estimator: Optional[Estimator]
    sampler: Sampler
    optimizer: Optimizer
    optimizer_n_circuit_evaluations: Optional[int]
    max_generations: Optional[int]
    max_circuit_evaluations: Optional[int]
    termination_criterion: Optional[EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion]
    random_seed: Optional[int]
    population_size: int
    randomize_initial_population_parameters: bool
    speciation_genetic_distance_threshold: int
    selection_alpha_penalty: float
    selection_beta_penalty: float
    parameter_search_probability: float
    topological_search_probability: float
    layer_removal_probability: float
    parallel_executor: Union[Client, ThreadPoolExecutor, None] = None
    mutually_exclusive_primitives: bool = True

    def __post_init__(self):
        if self.max_generations is None and self.max_circuit_evaluations is None and self.termination_criterion is None:
            raise ValueError(
                "At least one of the parameters max_generations, max_circuit_evaluations or"
                + "termination_criterion must not be None!"
            )
        if not 0 <= self.parameter_search_probability <= 1:
            raise ValueError("The parameter_search_probability must not exceed the range (0, 1)!")
        if not 0 <= self.topological_search_probability <= 1:
            raise ValueError("The topological_search_probability must not exceed the range (0, 1)!")
        if not 0 <= self.layer_removal_probability <= 1:
            raise ValueError("The layer_removal_probability must not exceed the range (0, 1)!")


class EVQEMinimumEigensolver(EvolvingAnsatzMinimumEigensolver):
    """Minimum eigensolver which uses the `EVQE` algorithm. For details see: https://arxiv.org/abs/1910.09694

    :param configuration: dataclass containing all configuration values for the EVQEMinimumEigensolver
    :type configuration: EVQEMinimumEigensolverConfiguration
    """

    def __init__(self, configuration: EVQEMinimumEigensolverConfiguration):
        self.random_generator: Random = Random(configuration.random_seed)

        population_initializer: Callable[[int], EVQEPopulation] = lambda n_qubits: EVQEPopulation.random_population(
            n_qubits=n_qubits,
            n_layers=1,
            n_individuals=configuration.population_size,
            randomize_parameter_values=configuration.randomize_initial_population_parameters,
            random_seed=new_random_seed(random_generator=self.random_generator),
        )

        evolutionary_operators: list[BaseEvolutionaryOperator] = [
            EVQELastLayerParameterSearch(
                mutation_probability=1,
                optimizer=configuration.optimizer,
                optimizer_n_circuit_evaluations=configuration.optimizer_n_circuit_evaluations,
                random_seed=new_random_seed(random_generator=self.random_generator),
            ),
            EVQESpeciation(
                genetic_distance_threshold=configuration.speciation_genetic_distance_threshold,
                random_seed=new_random_seed(random_generator=self.random_generator),
            ),
            EVQESelection(
                alpha_penalty=configuration.selection_alpha_penalty,
                beta_penalty=configuration.selection_beta_penalty,
                random_seed=new_random_seed(random_generator=self.random_generator),
            ),
            EVQEParameterSearch(
                mutation_probability=configuration.parameter_search_probability,
                optimizer=configuration.optimizer,
                optimizer_n_circuit_evaluations=configuration.optimizer_n_circuit_evaluations,
                random_seed=new_random_seed(random_generator=self.random_generator),
            ),
            EVQETopologicalSearch(
                mutation_probability=configuration.topological_search_probability,
                random_seed=new_random_seed(random_generator=self.random_generator),
            ),
            EVQELayerRemoval(
                mutation_probability=configuration.layer_removal_probability,
                random_seed=new_random_seed(random_generator=self.random_generator),
            ),
        ]

        parallel_executor: Union[Client, ThreadPoolExecutor]
        if configuration.parallel_executor is None:
            parallel_executor = ThreadPoolExecutor(max_workers=configuration.population_size)
        else:
            parallel_executor = configuration.parallel_executor

        config: EvolvingAnsatzMinimumEigensolverConfiguration = EvolvingAnsatzMinimumEigensolverConfiguration(
            population_initializer=population_initializer,
            evolutionary_operators=evolutionary_operators,
            estimator=configuration.estimator,
            sampler=configuration.sampler,
            max_generations=configuration.max_generations,
            max_circuit_evaluations=configuration.max_circuit_evaluations,
            termination_criterion=configuration.termination_criterion,
            parallel_executor=parallel_executor,
            mutually_exclusive_primitives=configuration.mutually_exclusive_primitives,
        )
        super().__init__(configuration=config)

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: Optional[ListOrDict[BaseOperator]] = None,
    ) -> EvolvingAnsatzMinimumEigensolverResult:
        return super().compute_minimum_eigenvalue(operator, aux_operators)

    def compute_minimum_function_value(
        self,
        operator: BitstringEvaluator,
        aux_operators: Optional[ListOrDict[BitstringEvaluator]] = None,
    ) -> EvolvingAnsatzMinimumEigensolverResult:
        return super().compute_minimum_function_value(operator, aux_operators)

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True
