# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from random import Random
from typing import Optional

from numpy import asarray, reshape, dtype
from numpy.typing import NDArray
from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.optimizers import Optimizer, OptimizerResult

from queasars.circuit_evaluation.circuit_evaluation import BaseCircuitEvaluator
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import (
    EVQEIndividual,
)


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
        )
        n_circuit_evaluations += needed_circuit_evaluations

    return current_individual, n_circuit_evaluations
