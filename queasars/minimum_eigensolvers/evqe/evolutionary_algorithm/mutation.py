# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from math import ceil
from random import Random
from typing import Optional

from dask.distributed import Future, wait
from numpy import asarray, reshape, dtype
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult

from queasars.circuit_evaluation.circuit_evaluation import BaseCircuitEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BaseEvolutionaryOperator, POP, OperatorContext
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

    print("optimize layer!")
    result: OptimizerResult = optimizer.minimize(
        fun=evaluation_callback,
        x0=asarray(parameter_values),
        bounds=[(None, None)] * len(parameter_values),
    )
    print("finished optimizing layer!")

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
        print(f"optimizing layer: {layer_to_optimize}, individual: {hash(individual)}")
        layer_indices.remove(layer_to_optimize)

        current_individual, needed_circuit_evaluations = optimize_layer_of_individual(
            individual=current_individual,
            layer_id=layer_to_optimize,
            evaluator=evaluator,
            optimizer=optimizer,
        )
        n_circuit_evaluations += needed_circuit_evaluations

    return current_individual, n_circuit_evaluations


class EVQELastLayerParameterSearch(BaseEvolutionaryOperator[EVQEPopulation]):
    """Mutation operator which optimizes the parameters for some EVQEIndividuals of the population for
    their last layer only. This operator empties the species_members information of the EVQEPopulation

    :param mutation_probability: with which an individual is mutated. Must be in the range (0, 1)
    :type mutation_probability: float
    :param optimizer: qiskit optimizer which is used to optimize the parameters
    :type optimizer: Optimizer
    :param expected_circuit_evaluations_per_optimizer_run: amount of circuit evaluations the optimizer needs per
        optimization run. This can be an estimate, or a value, which was set as a setting for the optimizer.
        If no estimate can be made this value should be set to None
    :type expected_circuit_evaluations_per_optimizer_run: Optional[int]
    :param random_seed: seed value to control randomness
    :type random_seed: int
    """

    def __init__(
        self,
        mutation_probability: float,
        optimizer: Optimizer,
        expected_circuit_evaluations_per_optimizer_run: Optional[int],
        random_seed: Optional[int] = None,
    ):
        self.mutation_probability: float = mutation_probability
        self.optimizer: Optimizer = optimizer
        self.expected_circuit_evaluations_per_optimizer_run: Optional[
            int
        ] = expected_circuit_evaluations_per_optimizer_run
        self.random_generator: Random = Random(random_seed)

    def apply_operator(self, population: EVQEPopulation, operator_context: OperatorContext) -> EVQEPopulation:
        mutated_individuals: dict[int, Future] = {}
        total_circuit_evaluations: int = 0

        for i, individual in enumerate(population.individuals):
            if self.random_generator.random() <= self.mutation_probability:
                last_layer_id = len(individual.layers) - 1
                mutated_individuals[i] = operator_context.dask_client.submit(
                    optimize_layer_of_individual,
                    individual,
                    last_layer_id,
                    operator_context.circuit_evaluator,
                    self.optimizer,
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
            species_members={},
        )

    def get_n_expected_circuit_evaluations(self, population: POP) -> Optional[int]:
        if self.expected_circuit_evaluations_per_optimizer_run is not None:
            expectation_value: float = (
                self.mutation_probability
                * len(population.individuals)
                * self.expected_circuit_evaluations_per_optimizer_run
            )
            return ceil(expectation_value)
        return None


class EVQEParameterSearch(BaseEvolutionaryOperator[EVQEPopulation]):
    """Mutation operator which optimizes the parameters for some EVQEIndividuals in the population layer by layer in
    a random order. This operator empties the species_members information of the EVQEPopulation

    :param mutation_probability: with which an individual is mutated. Must be in the range (0, 1)
    :type mutation_probability: float
    :param optimizer: qiskit optimizer which is used to optimize the parameters
    :type optimizer: Optimizer
    :param expected_circuit_evaluations_per_optimizer_run: amount of circuit evaluations the optimizer needs per
        optimization run. This can be an estimate, or a value, which was set as a setting for the optimizer.
        If no estimate can be made this value should be set to None
    :type expected_circuit_evaluations_per_optimizer_run: Optional[int]
    :param random_seed: seed value to control randomness
    :type random_seed: int
    """

    def __init__(
        self,
        mutation_probability: float,
        optimizer: Optimizer,
        expected_circuit_evaluations_per_optimizer_run: Optional[int],
        random_seed: Optional[int] = None,
    ):
        """Constructor method."""
        self.mutation_probability: float = mutation_probability
        self.optimizer: Optimizer = optimizer
        self.expected_circuit_evaluations_per_optimizer_run: Optional[
            int
        ] = expected_circuit_evaluations_per_optimizer_run
        self.random_generator: Random = Random(random_seed)

    def apply_operator(self, population: EVQEPopulation, operator_context: OperatorContext) -> EVQEPopulation:
        mutated_individuals: dict[int, Future] = {}
        total_circuit_evaluations: int = 0

        for i, individual in enumerate(population.individuals):
            if self.random_generator.random() <= self.mutation_probability:
                seed = new_random_seed(self.random_generator)
                mutated_individuals[i] = operator_context.dask_client.submit(
                    optimize_all_parameters_of_individual,
                    individual,
                    operator_context.circuit_evaluator,
                    self.optimizer,
                    seed,
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
            species_members={},
        )

    def get_n_expected_circuit_evaluations(self, population: EVQEPopulation) -> Optional[int]:
        average_n_layers: float = sum(len(individual.layers) for individual in population.individuals) / len(
            population.individuals
        )
        if self.expected_circuit_evaluations_per_optimizer_run is not None:
            expectation_value: float = (
                self.mutation_probability
                * len(population.individuals)
                * average_n_layers
                * self.expected_circuit_evaluations_per_optimizer_run
            )
            return ceil(expectation_value)
        return None


class EVQETopologicalSearch(BaseEvolutionaryOperator[EVQEPopulation]):
    """Mutation operator which adds one random circuit layer to some individuals of the population.
    This operator empties the species_members information of the EVQEPopulation

    :param mutation_probability: with which an individual is mutated. Must be in the range (0, 1)
    :type mutation_probability: float
    :param random_seed: integer value to control randomness
    :type random_seed: Optional[int]
    """

    def __init__(self, mutation_probability: float, random_seed: Optional[int] = None):
        self.mutation_probability: float = mutation_probability
        self.random_generator: Random = Random(random_seed)

    def apply_operator(self, population: EVQEPopulation, operator_context: OperatorContext) -> EVQEPopulation:
        new_individuals: list[EVQEIndividual] = list(population.individuals)

        for i, individual in enumerate(population.individuals):
            if self.random_generator.random() <= self.mutation_probability:
                new_individuals[i] = EVQEIndividual.add_random_layers(
                    individual=individual,
                    n_layers=1,
                    randomize_parameter_values=False,
                    random_seed=new_random_seed(self.random_generator),
                )

        return EVQEPopulation(
            individuals=tuple(new_individuals),
            species_representatives=population.species_representatives,
            species_members={},
        )

    def get_n_expected_circuit_evaluations(self, population: EVQEPopulation) -> Optional[int]:
        return 0
