# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import abstractmethod
from dataclasses import dataclass
from math import ceil
from random import Random
from typing import Optional, TypeAlias, Callable

from dask.distributed import Future, wait
from numpy import asarray, reshape, dtype
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult

from queasars.circuit_evaluation.circuit_evaluation import BaseCircuitEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BaseEvolutionaryOperator, OperatorContext
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import (
    EVQEIndividual,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.population import EVQEPopulation
from queasars.utility.random import new_random_seed


def optimize_layer_of_individual(
    individual: EVQEIndividual,
    layer_id: int,
    evaluator: BaseCircuitEvaluator,
    optimizer: Optimizer,
    random_seed: Optional[int],
) -> tuple[EVQEIndividual, int]:
    """
    Optimizes the parameter values of one given circuit layer of the given individual. Returns a new
    individual with the optimized parameter values as well as the amount of circuit evaluations used.

    :arg individual: for which to optimize the parameter values
    :type individual: EVQEIndividual
    :arg layer_id: index of the layer for which the parameter values shall be optimized
    :type layer_id: int
    :arg evaluator: circuit evaluator used to get the objective value for the optimizer
    :type evaluator: BaseCircuitEvaluator
    :arg optimizer: qiskit optimizer used to optimize the parameter values
    :type optimizer: Optimizer
    :arg random_seed: integer value to control randomness
    :type random_seed: int
    :return: a new individual with optimized parameter values for the given layer
    :rtype: tuple[EVQEIndividual, int]
    """
    parameterized_circuit: QuantumCircuit = individual.get_partially_parameterized_quantum_circuit(
        parameterized_layers={layer_id}
    )
    parameter_values: tuple[float, ...] = individual.get_layer_parameter_values(layer_id=layer_id)
    n_parameters: int = len(parameter_values)

    def evaluation_callback(parameter_values: NDArray) -> NDArray | float:
        parameters: list[list[float]] = reshape(parameter_values, (-1, n_parameters)).tolist()
        batch_size: int = len(parameters)

        evaluation_result: list[float] = evaluator.evaluate_circuits(
            circuits=[parameterized_circuit] * batch_size,
            parameter_values=parameters,
        )

        if len(evaluation_result) == 1:
            return evaluation_result[0]

        return asarray(evaluation_result, dtype=dtype("float64"))

    result: OptimizerResult = optimizer.minimize(
        fun=evaluation_callback,
        x0=asarray(parameter_values),
        bounds=[(None, None)] * len(parameter_values),
    )

    result_parameter_values: list[float] = list(result.x)
    n_circuit_evaluations: int = result.nfev

    return (
        EVQEIndividual.change_layer_parameter_values(individual, layer_id, tuple(result_parameter_values)),
        n_circuit_evaluations,
    )


def optimize_all_parameters_of_individual(
    individual: EVQEIndividual,
    evaluator: BaseCircuitEvaluator,
    optimizer: Optimizer,
    random_seed: Optional[int] = None,
) -> tuple[EVQEIndividual, int]:
    """
    Optimizes all parameters of an individual. The parameters are optimized one layer at a time, with the
    ordering being random. A new individual with the optimized parameters is returned as well as the number
    of the circuit evaluations which were needed

    :arg individual: for which to optimize the parameter values
    :type individual: EVQEIndividual
    :arg evaluator: circuit evaluator used to get the objective value for the optimizer
    :type evaluator: BaseCircuitEvaluator
    :arg optimizer: qiskit optimizer used to optimize the parameter values
    :type optimizer: Optimizer
    :arg random_seed: random seed used for the random layer ordering
    :type random_seed: Optional[int]
    :return: a new individual with optimized parameters and the number of the circuit evaluations which were used
    :rtype: tuple[EVQEIndividual, int]
    """
    randomizer: Random = Random(random_seed)
    layer_indices: list[int] = list(range(0, len(individual.layers)))
    n_circuit_evaluations: int = 0
    current_individual: EVQEIndividual = individual

    while len(layer_indices) > 0:
        layer_to_optimize: int = randomizer.choice(layer_indices)
        layer_indices.remove(layer_to_optimize)

        current_individual, needed_circuit_evaluations = optimize_layer_of_individual(
            individual=current_individual,
            layer_id=layer_to_optimize,
            evaluator=evaluator,
            optimizer=optimizer,
            random_seed=new_random_seed(randomizer),
        )
        n_circuit_evaluations += needed_circuit_evaluations

    return current_individual, n_circuit_evaluations


def remove_random_layers_from_individual(individual: EVQEIndividual, random_seed: Optional[int]) -> EVQEIndividual:
    """
    Returns a new individual which is based on the given individual with its last few layers randomly removed

    :arg individual: from which to remove layers
    :type individual: EVQEIndividual
    :arg random_seed: integer value to control randomness
    :type random_seed: Optional[int]
    :return: the new individual
    :rtype: EVQEIndividual
    """
    random_generator: Random = Random(random_seed)
    n_layers_to_remove: int = random_generator.randrange(1, len(individual.layers))
    return EVQEIndividual.remove_layers(individual=individual, n_layers=n_layers_to_remove)


MutationFunction: TypeAlias = Callable[
    [EVQEIndividual, BaseCircuitEvaluator, Optimizer, Optional[int]], tuple[EVQEIndividual, int]
]


@dataclass
class EVQEOperatorContext(OperatorContext):
    """Dataclass containing additional references needed by operators of the EVQE algorithm

    :param circuit_evaluator: CircuitEvaluator used to get the expectation value of the circuits (Individuals)
    :type circuit_evaluator: BaseCircuitEvaluator
    :param result_callback: Callback function to report results from evaluating a population
    :type result_callback: Callable[[BasePopulationEvaluationResult], None]
    :param circuit_evaluation_count_callback: Callback functions to report the number of circuit evaluations
        used by an operator.
    :type circuit_evaluation_count_callback: Callable[[Int], None]
    :param dask_client: Dask client to use for task parallelization
    :type: Client
    :param optimizer: qiskit optimizer used to optimize the parameter values
    :type optimizer: Optimizer
    :param optimizer_n_circuit_evaluations: amount of circuit evaluations needed by the optimizer, None if unknown
    :type optimizer_n_circuit_evaluations: Optional[int]
    """

    optimizer: Optimizer
    optimizer_n_circuit_evaluations: Optional[int]


class BaseEVQEMutationOperator(BaseEvolutionaryOperator[EVQEPopulation, EVQEOperatorContext]):
    """Base class for mutation operators for the EVQE algorithm.
    This operator empties the species_members information of the EVQEPopulation

    :arg mutation_function: function to be applied to the individuals.
        Takes an EVQEIndividual, BaseCircuitEvaluator, Optimizer, and a random seed (int) and returns a tuple
        of the new individual and the amount of circuit evaluations (int) which were needed
    :type mutation_function: MutationFunction
    :arg mutation_probability: with which the mutation_function is applied to an individual
    :type mutation_probability: float
    :arg random_seed: integer value to control randomness
    :type random_seed: int
    """

    def __init__(
        self,
        mutation_function: MutationFunction,
        mutation_probability: float,
        random_seed: Optional[int] = None,
    ):
        """Constructor method"""
        self.mutation_function: MutationFunction = mutation_function
        self.mutation_probability: float = mutation_probability
        self.random_generator: Random = Random(random_seed)

    def apply_operator(self, population: EVQEPopulation, operator_context: EVQEOperatorContext) -> EVQEPopulation:
        mutated_individuals: dict[int, Future] = {}
        total_circuit_evaluations: int = 0

        for i, individual in enumerate(population.individuals):
            if self.random_generator.random() <= self.mutation_probability:
                mutated_individuals[i] = operator_context.dask_client.submit(
                    self.mutation_function,
                    individual,
                    operator_context.circuit_evaluator,
                    operator_context.optimizer,
                    new_random_seed(random_generator=self.random_generator),
                )

        wait(mutated_individuals.values())

        new_individuals = list(population.individuals)
        for i, result in mutated_individuals.items():
            new_individual, circuit_evaluations = result.result()
            new_individuals[i] = new_individual
            total_circuit_evaluations += circuit_evaluations

        del mutated_individuals

        operator_context.circuit_evaluation_count_callback(total_circuit_evaluations)

        return EVQEPopulation(
            individuals=tuple(new_individuals),
            species_representatives=population.species_representatives,
            species_members=None,
            species_membership=None,
        )

    @abstractmethod
    def get_n_expected_circuit_evaluations(
        self, population: EVQEPopulation, operator_context: EVQEOperatorContext
    ) -> Optional[int]:
        pass


class EVQELastLayerParameterSearch(BaseEVQEMutationOperator):
    """Mutation operator which optimizes the parameters for some EVQEIndividuals of the population for
    their last layer only. This operator empties the species_members information of the EVQEPopulation

    :param mutation_probability: with which an individual is mutated. Must be in the range (0, 1)
    :type mutation_probability: float
    :param random_seed: seed value to control randomness
    :type random_seed: int
    """

    def __init__(
        self,
        mutation_probability: float,
        random_seed: Optional[int] = None,
    ):
        mutation_function: MutationFunction = (
            lambda individual, evaluator, optimizer, seed: optimize_layer_of_individual(
                individual=individual, layer_id=-1, evaluator=evaluator, optimizer=optimizer, random_seed=seed
            )
        )
        super().__init__(
            mutation_function=mutation_function, mutation_probability=mutation_probability, random_seed=random_seed
        )

    def apply_operator(self, population: EVQEPopulation, operator_context: EVQEOperatorContext) -> EVQEPopulation:
        return super().apply_operator(population=population, operator_context=operator_context)

    def get_n_expected_circuit_evaluations(
        self, population: EVQEPopulation, operator_context: EVQEOperatorContext
    ) -> Optional[int]:
        if operator_context.optimizer_n_circuit_evaluations is not None:
            expectation_value: float = (
                self.mutation_probability
                * len(population.individuals)
                * operator_context.optimizer_n_circuit_evaluations
            )
            return ceil(expectation_value)
        return None


class EVQEParameterSearch(BaseEVQEMutationOperator):
    """Mutation operator which optimizes the parameters for some EVQEIndividuals in the population layer by layer in
    a random order. This operator empties the species_members information of the EVQEPopulation

    :param mutation_probability: with which an individual is mutated. Must be in the range (0, 1)
    :type mutation_probability: float
    :param random_seed: seed value to control randomness
    :type random_seed: int
    """

    def __init__(
        self,
        mutation_probability: float,
        random_seed: Optional[int] = None,
    ):
        """Constructor method"""
        mutation_function: MutationFunction = optimize_all_parameters_of_individual
        super().__init__(
            mutation_function=mutation_function, mutation_probability=mutation_probability, random_seed=random_seed
        )

    def apply_operator(self, population: EVQEPopulation, operator_context: EVQEOperatorContext) -> EVQEPopulation:
        return super().apply_operator(population=population, operator_context=operator_context)

    def get_n_expected_circuit_evaluations(
        self, population: EVQEPopulation, operator_context: EVQEOperatorContext
    ) -> Optional[int]:
        average_n_layers: float = sum(len(individual.layers) for individual in population.individuals) / len(
            population.individuals
        )
        if operator_context.optimizer_n_circuit_evaluations is not None:
            expectation_value: float = (
                self.mutation_probability
                * len(population.individuals)
                * average_n_layers
                * operator_context.optimizer_n_circuit_evaluations
            )
            return ceil(expectation_value)
        return None


class EVQETopologicalSearch(BaseEVQEMutationOperator):
    """Mutation operator which adds one random circuit layer to some individuals of the population.
    This operator empties the species_members information of the EVQEPopulation

    :param mutation_probability: with which an individual is mutated. Must be in the range (0, 1)
    :type mutation_probability: float
    :param random_seed: integer value to control randomness
    :type random_seed: Optional[int]
    """

    def __init__(self, mutation_probability: float, random_seed: Optional[int] = None):
        mutation_function: MutationFunction = lambda individual, evaluator, optimizer, seed: (
            EVQEIndividual.add_random_layers(individual=individual, n_layers=1, randomize_parameter_values=False),
            0,
        )
        super().__init__(
            mutation_function=mutation_function, mutation_probability=mutation_probability, random_seed=random_seed
        )

    def apply_operator(self, population: EVQEPopulation, operator_context: EVQEOperatorContext) -> EVQEPopulation:
        return super().apply_operator(population=population, operator_context=operator_context)

    def get_n_expected_circuit_evaluations(
        self, population: EVQEPopulation, operator_context: EVQEOperatorContext
    ) -> Optional[int]:
        return 0


class EVQELayerRemoval(BaseEVQEMutationOperator):
    """Mutation operator which removes a random amount of circuit layers from some individuals of the population.
    This operator empties the species_members information of the EVQEPopulation

    :param mutation_probability: with which an individual is mutated. Must be in the range (0, 1)
    :type mutation_probability: float
    :param random_seed: integer value to control randomness
    :type random_seed: Optional[int]"""

    def __init__(self, mutation_probability: float, random_seed: Optional[int]):
        mutation_function: MutationFunction = lambda individual, evaluator, optimizer, seed: (
            remove_random_layers_from_individual(individual=individual, random_seed=seed),
            0,
        )
        super().__init__(
            mutation_function=mutation_function, mutation_probability=mutation_probability, random_seed=random_seed
        )

    def apply_operator(self, population: EVQEPopulation, operator_context: EVQEOperatorContext) -> EVQEPopulation:
        return super().apply_operator(population=population, operator_context=operator_context)

    def get_n_expected_circuit_evaluations(
        self, population: EVQEPopulation, operator_context: EVQEOperatorContext
    ) -> Optional[int]:
        return 0
