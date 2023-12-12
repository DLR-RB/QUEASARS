# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler, EstimatorResult
from qiskit.quantum_info.operators.base_operator import BaseOperator

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator


class BaseCircuitEvaluator(ABC):
    """Abstract class to allow a seamless exchange of circuit
    evaluation methods in QUEASARS eigensolvers"""

    @abstractmethod
    def evaluate_circuits(self, circuits: list[QuantumCircuit], parameter_values: list[list[float]]) -> list[float]:
        """Evaluates a list of parameterized QuantumCircuits for a list of parameter_vales. The circuit
        at index i is evaluated for parameter_values list at index i. The matching result is at index i of
        the returned list

        :arg circuits: Quantum circuit to evaluate, must be parameterized
        :type circuits: list[QuantumCircuit]
        :arg parameter_values: Parameters for which the parameterized circuits shall be evaluated
        :type parameter_values: list[list[float]]
        :raises CircuitEvaluatorException: If the circuit evaluation fails for any reason
        :return: The list of the resulting floating point number
        :rtype: list[list[float]]"""


class CircuitEvaluatorException(Exception):
    """Class for exceptions caused during the evaluation of quantum circuits"""


class OperatorCircuitEvaluator(BaseCircuitEvaluator):
    """Class which evaluates quantum circuits by estimating the eigenvalue of the circuit for a given operator

    :param estimator: Estimator primitive used for estimating the circuit's eigenvalue
    :type estimator: BaseEstimator
    :param operator: Operator for which the eigenvalue is estimated
    :type operator: BaseOperator
    """

    def __init__(self, estimator: BaseEstimator, operator: BaseOperator):
        """Constructor Method"""
        self.estimator: BaseEstimator = estimator
        self.operator: BaseOperator = operator

    def evaluate_circuits(self, circuits: list[QuantumCircuit], parameter_values: list[list[float]]) -> list[float]:
        evaluation_result: EstimatorResult = self.estimator.run(
            circuits=circuits, observables=[self.operator] * len(circuits), parameter_values=parameter_values
        ).result()
        result_list: list[float] = list(evaluation_result.values)
        return result_list


class BitstringCircuitEvaluator(BaseCircuitEvaluator):
    """Class which evaluates quantum circuits by calculating an expectation value over the circuit's sampled
    measurement distribution using the :class:`queasars.circuit_evaluation.bitstring_evaluation.BitstringEvaluator`
    to assign floating point values to individual measurements

    :param sampler: Sampler primitive to retrieve measurement distributions with
    :type sampler: BaseSampler
    :param bitstring_evaluator: Evaluation function used to evaluate individual measurements
    :type bitstring_evaluator: BitstringEvaluator
    """

    def __init__(self, sampler: BaseSampler, bitstring_evaluator: BitstringEvaluator):
        """Constructor Method"""

    def evaluate_circuits(self, circuits: list[QuantumCircuit], parameter_values: list[list[float]]) -> list[float]:
        raise NotImplementedError
