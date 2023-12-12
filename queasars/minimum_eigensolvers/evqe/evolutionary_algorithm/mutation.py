# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from qiskit_algorithms.optimizers import Optimizer, OptimizerResult
from numpy import asarray, reshape, dtype
from numpy.typing import NDArray

from queasars.circuit_evaluation.circuit_evaluation import BaseCircuitEvaluator
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import (
    EVQEIndividual,
)


def optimize_layer_of_individual(
    individual: EVQEIndividual,
    layer_id: int,
    evaluator: BaseCircuitEvaluator,
    optimizer: Optimizer,
) -> EVQEIndividual:
    """
    Optimizes the parameter values of one given circuit layer of the given individual and returns a new
    individual with the optimized parameter values

    :arg individual: for which to optimize the parameter values
    :type individual: EVQEIndividual
    :arg layer_id: index of the layer for which the parameter values shall be optimized
    :type layer_id: int
    :arg evaluator: circuit evaluator used to get the objective value for the optimizer
    :type evaluator: BaseCircuitEvaluator
    :arg optimizer: qiskit optimizer used to optimize the parameter values
    :type optimizer: Optimizer
    :return: a new individual with optimized parameter values for the given layer
    :rtype: EVQEIndividual
    """
    parameterized_circuit = individual.get_partially_parameterized_quantum_circuit(parameterized_layers={layer_id})
    parameter_values = individual.get_layer_parameter_values(layer_id=layer_id)
    n_parameters = len(parameter_values)

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
    return EVQEIndividual.change_layer_parameter_values(individual, layer_id, tuple(result_parameter_values))
