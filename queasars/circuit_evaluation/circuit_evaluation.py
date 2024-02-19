# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from typing import Union

from numpy import real

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler, EstimatorResult, SamplerResult, PrimitiveJob
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import QuasiDistribution

from qiskit_algorithms.minimum_eigensolvers.diagonal_estimator import _DiagonalEstimator

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

    @property
    @abstractmethod
    def n_qubits(self) -> int:
        """Get the size of the quantum circuits (in terms of qubits), which this CircuitEvaluator can evaluate

        :return: the amount of qubits
        :rtype: int
        """


class CircuitEvaluatorException(Exception):
    """Class for exceptions caused during the evaluation of quantum circuits"""


class OperatorCircuitEvaluator(BaseCircuitEvaluator):
    """Class which evaluates quantum circuits by estimating the eigenvalue of the circuit for a given operator

    :param qiskit_primitive: Qiskit primitive used for estimating the circuit's eigenvalue.
        Must be an Estimator or a Sampler
    :type qiskit_primitive: BaseEstimator | BaseSampler
    :param operator: Operator for which the eigenvalue is estimated. If the qiskit_primitive is a sampler it
        must be diagonal
    :type operator: BaseOperator
    """

    def __init__(
        self,
        qiskit_primitive: Union[BaseEstimator[PrimitiveJob[EstimatorResult]], BaseSampler[PrimitiveJob[SamplerResult]]],
        operator: BaseOperator,
    ):
        """Constructor Method"""
        self.estimator: BaseEstimator[PrimitiveJob[EstimatorResult]]
        if isinstance(qiskit_primitive, BaseEstimator):
            self.estimator = qiskit_primitive
        self.measure: bool = False
        if isinstance(qiskit_primitive, BaseSampler):
            self.measure = True
            self.estimator = _DiagonalEstimator(sampler=qiskit_primitive)
        self.operator: BaseOperator = operator

    def evaluate_circuits(self, circuits: list[QuantumCircuit], parameter_values: list[list[float]]) -> list[float]:
        if self.measure:
            circuits = [circuit.measure_all(inplace=False) for circuit in circuits]
        evaluation_result: EstimatorResult = self.estimator.run(
            circuits=circuits, observables=[self.operator] * len(circuits), parameter_values=parameter_values
        ).result()
        result_list: list[float] = list(real(evaluation_result.values))
        return result_list

    @property
    def n_qubits(self) -> int:
        return self.operator.num_qubits


class BitstringCircuitEvaluator(BaseCircuitEvaluator):
    """Class which evaluates quantum circuits by calculating an expectation value over the circuit's sampled
    measurement distribution using the :class:`queasars.circuit_evaluation.bitstring_evaluation.BitstringEvaluator`
    to assign floating point values to individual measurements

    :param sampler: Sampler primitive to retrieve measurement distributions with
    :type sampler: BaseSampler
    :param bitstring_evaluator: Evaluation function used to evaluate individual measurements
    :type bitstring_evaluator: BitstringEvaluator
    """

    def __init__(self, sampler: BaseSampler[PrimitiveJob[SamplerResult]], bitstring_evaluator: BitstringEvaluator):
        """Constructor Method"""
        self.sampler: BaseSampler[PrimitiveJob[SamplerResult]] = sampler
        self.bitstring_evaluator: BitstringEvaluator = bitstring_evaluator

    def evaluate_circuits(self, circuits: list[QuantumCircuit], parameter_values: list[list[float]]) -> list[float]:
        circuits = [circuit.measure_all(inplace=False) for circuit in circuits]
        evaluation_result: SamplerResult = self.sampler.run(
            circuits=circuits, parameter_values=parameter_values
        ).result()
        quasi_distributions: list[QuasiDistribution] = evaluation_result.quasi_dists
        expectation_values: list[float] = []
        for distribution in quasi_distributions:
            binary_distribution = distribution.binary_probabilities(num_bits=self.bitstring_evaluator.input_length)
            expectation_values.append(
                sum(
                    probability * self.bitstring_evaluator.evaluate_bitstring(bitstring)
                    for bitstring, probability in binary_distribution.items()
                )
            )
        return expectation_values

    @property
    def n_qubits(self) -> int:
        return self.bitstring_evaluator.input_length
