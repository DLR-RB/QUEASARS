# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Any, Union

from numpy import isclose
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import QuasiDistribution, ProbDistribution, sampled_expectation_value
from qiskit_algorithms.minimum_eigensolvers.diagonal_estimator import _evaluate_sparsepauli

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator


def _get_expectation(state_list: list[tuple[Any, float, float]], alpha: float):

    if not isclose(alpha, 1):
        state_list = sorted(state_list, key=lambda x: x[2])

    gathered: float = 0
    expectation: float = 0
    for _, probability, value in state_list:

        probability = min(alpha - gathered, probability)
        expectation += probability * value
        gathered += probability

        if isclose(gathered, alpha):
            break

    expectation = expectation / alpha

    return expectation


def get_expectation_with_operator(
    measurement_distribution: Union[QuasiDistribution, ProbDistribution], operator: SparsePauliOp, alpha: float = 1
) -> float:
    """
    Calculates the expectation value of a quantum state, characterized by a probability distribution of
    z-basis measurement results. The operator with respect to which the expectation is calculated needs to
    be a diagonal SparsePauliOp. If desired, this expectation can be calculated over only the lower alpha
    tail of the probability distribution. In that case, this calculation equals the CVaR expectation as
    outlined in https://quantum-journal.org/papers/q-2020-04-20-256/

    :arg measurement_distribution: distribution of measurements from which to derive the expectation value
    :type measurement_distribution: Union[QuasiDistribution, ProbDistribution]
    :arg operator: operator with respect to which the expectation value is calculated
    :type operator: SparsePauliOp
    :arg alpha: specifies the lower tail of the distribution which is taken into account during the
        expectation value calculation. Must be in the range (0, 1]. By default, alpha is 1. In that case
        the normal expectation value over the whole distribution is calculated
    :return:
    """

    if alpha <= 0 or 1 < alpha:
        raise ValueError("alpha must be in the range (0, 1]!")

    measurements: list[tuple[int, float]] = list(measurement_distribution.items())

    if isclose(alpha, 1):
        return sampled_expectation_value(dist=measurement_distribution, oper=operator)

    # TODO: Find a performant way to calculate state evaluations without _evaluate_sparse_pauli
    evaluations: list[tuple[int, float, float]] = [
        (state, probability, _evaluate_sparsepauli(state, operator).real) for state, probability in measurements
    ]
    evaluations = sorted(evaluations, key=lambda x: x[2])

    return _get_expectation(state_list=evaluations, alpha=alpha)


def get_expectation_with_bitstring_evaluator(
    measurement_distribution: Union[QuasiDistribution, ProbDistribution],
    bitstring_evaluator: BitstringEvaluator,
    alpha: float = 1,
) -> float:
    """
    Calculates the expectation value of a quantum state, characterized by a probability distribution of
    z-basis measurement results. If desired, this expectation can be calculated over only the lower alpha
    tail of the probability distribution. In that case, this calculation equals the CVaR expectation as
    outlined in https://quantum-journal.org/papers/q-2020-04-20-256/

    :arg measurement_distribution: distribution of measurements from which to derive the expectation value
    :type measurement_distribution: Union[QuasiDistribution, ProbDistribution]
    :arg bitstring_evaluator: evaluator, which assigns a value to individual bitstring measurements
    :type bitstring_evaluator: BitstringEvaluator
    :arg alpha: specifies the lower tail of the distribution which is taken into account during the
        expectation value calculation. Must be in the range (0, 1]. By default, alpha is 1. In that case
        the normal expectation value over the whole distribution is calculated
    :return:
    """

    if alpha <= 0 or 1 < alpha:
        raise ValueError("alpha must be in the range (0, 1]!")

    measurements: list[tuple[str, float]] = list(measurement_distribution.binary_probabilities().items())

    evaluations: list[tuple[str, float, float]] = [
        (state, probability, bitstring_evaluator.evaluate_bitstring(bitstring=state))
        for state, probability in measurements
    ]

    return _get_expectation(state_list=evaluations, alpha=alpha)
