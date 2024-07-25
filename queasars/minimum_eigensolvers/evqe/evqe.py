# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from random import Random
from typing import Callable, Optional, Union

from dask.distributed import Client
from qiskit.transpiler import PassManager
from qiskit_algorithms.optimizers import Optimizer

from queasars.circuit_evaluation.configured_primitives import ConfiguredEstimatorV2, ConfiguredSamplerV2
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BaseEvolutionaryOperator
from queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver import (
    EvolvingAnsatzMinimumEigensolver,
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

    :param configured_estimator: Configured EstimatorV2 primitive used to estimate the circuit's eigenvalue.
        If none is provided for that purpose, the sampler is used instead. If a dask Client is used as the
        parallel_executor, the Estimator needs to be serializable by dask, otherwise the computation will fail
    :type configured_estimator: Optional[ConfiguredEstimatorV2]
    :param configured_sampler: Sampler primitive used to measure the circuits QuasiDistribution.
        If a dask Client is used as the parallel_executor, the Sampler needs to be serializable by dask,
        otherwise the computation will fail
    :param pass_manager: A qiskit PassManager which specifies the transpilation procedure. If no pass_manager is given,
        a preset passmanager with optimization level 0 and no information on the backend is used. When running on
        real quantum hardware, the pass_manager must be user_configured to fit the backend
    :type pass_manager: PassManager
    :type configured_sampler: ConfiguredSamplerV2
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
    :param n_initial_layers: number of layers with which the individuals in the initial population are initialized.
        By default, this is set to 1. This should only be increased, if randomize_initial_population_parameters
        is set to True, or the parameter_search_probability is high. Otherwise, the added layers may start as an
        identity operator and only seldom be optimized, minimizing their effect.
    :type n_initial_layers: int
    :param use_tournament_selection: indicates whether to use tournament selection. By default, this is
        set to False. In that case, roulette wheel selection is used. Should be true, if the measured expectation
        values can be negative.
    :type use_tournament_selection: bool
    :param tournament_size: indicates the size of the tournaments used. This can be in the range [1, population_size].
        It cannot be None, if use_tournament_selection is set to True. A tournament_size of 1 yields random selection,
        with increasing tournament selection sizes increasing the selection pressure.
    :type tournament_size: int
    :param randomize_initial_population_parameters: Determines whether the parameter values of the individuals in
        the first population shall be initialized randomly or at 0. By default, the parameter values in the
        initial population are initialized randomly
    :type randomize_initial_population_parameters: bool
    :type parallel_executor: Union[Client, ThreadPoolExecutor, None]
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
        For safety reasons this is enabled by default. If the sampler and estimator are threadsafe or the dask client
        is configured with only one thread per process, disabling this option may lead to performance improvements
    :type mutually_exclusive_primitives: bool
    """

    configured_estimator: Optional[ConfiguredEstimatorV2]
    configured_sampler: ConfiguredSamplerV2
    pass_manager: Optional[PassManager]
    optimizer: Optimizer
    optimizer_n_circuit_evaluations: Optional[int]
    max_generations: Optional[int]
    max_circuit_evaluations: Optional[int]
    termination_criterion: Optional[EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion]
    random_seed: Optional[int]
    population_size: int
    speciation_genetic_distance_threshold: int
    selection_alpha_penalty: float
    selection_beta_penalty: float
    parameter_search_probability: float
    topological_search_probability: float
    layer_removal_probability: float
    n_initial_layers: int = 1
    use_tournament_selection: bool = False
    tournament_size: Optional[int] = None
    randomize_initial_population_parameters: bool = True
    parallel_executor: Union[Client, ThreadPoolExecutor, None] = None
    distribution_alpha_tail: float = 1
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
        if self.n_initial_layers < 1:
            raise ValueError(
                "The number of initial layers for each individual "
                + f"of the population must be at least 1! But it was {self.n_initial_layers}!"
            )
        if self.use_tournament_selection and self.tournament_size is None:
            raise ValueError("To use tournament_selection, a tournament_size must be specified! It cannot be None!")
        if self.use_tournament_selection and not 1 <= self.tournament_size:
            raise ValueError(f"The tournament_size cannot be smaller than 1!, but it was {self.tournament_size}!")
        if self.use_tournament_selection and self.population_size < self.tournament_size:
            raise ValueError(
                f"The tournament_size cannot be larger than the size of the population ({self.population_size})! \n"
                + f"Yet the tournament_size is {self.tournament_size}!"
            )


class EVQEMinimumEigensolver(EvolvingAnsatzMinimumEigensolver):
    """Minimum eigensolver which uses the `EVQE` algorithm. For details see: https://arxiv.org/abs/1910.09694

    :param configuration: dataclass containing all configuration values for the EVQEMinimumEigensolver
    :type configuration: EVQEMinimumEigensolverConfiguration
    """

    def __init__(self, configuration: EVQEMinimumEigensolverConfiguration):
        self.random_generator: Random = Random(configuration.random_seed)

        population_initializer: Callable[[int], EVQEPopulation] = lambda n_qubits: EVQEPopulation.random_population(
            n_qubits=n_qubits,
            n_layers=configuration.n_initial_layers,
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
                use_tournament_selection=configuration.use_tournament_selection,
                tournament_size=configuration.tournament_size,
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
            configured_estimator=configuration.configured_estimator,
            configured_sampler=configuration.configured_sampler,
            pass_manager=configuration.pass_manager,
            max_generations=configuration.max_generations,
            max_circuit_evaluations=configuration.max_circuit_evaluations,
            termination_criterion=configuration.termination_criterion,
            parallel_executor=parallel_executor,
            mutually_exclusive_primitives=configuration.mutually_exclusive_primitives,
            distribution_alpha_tail=configuration.distribution_alpha_tail,
        )
        super().__init__(configuration=config)

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True
