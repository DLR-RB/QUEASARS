# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.quantum_info.operators.base_operator import BaseOperator

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator


class BaseCircuitEvaluator(ABC):
    """Abstract class to allow a seamless exchange of circuit
    evaluation methods in QUEASARS eigensolvers"""

    @abstractmethod
    def evaluate_circuit(self, circuit: QuantumCircuit, angles: list[float]) -> float:
        """Evaluates a parameterized quantum circuit, for a given list of rotation angles.

        :arg circuit: Quantum circuit to evaluate, must be parameterized
        :type circuit: QuantumCircuit
        :arg angles: Angles for which the parameterized circuit shall be evaluated
        :type angles: list[float]
        :raises CircuitEvaluatorException: If the circuit evaluation fails for any reason
        :return: The resulting floating point number"""


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

    def evaluate_circuit(self, circuit: QuantumCircuit, angles: list[float]) -> float:
        raise NotImplementedError


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

    def evaluate_circuit(self, circuit: QuantumCircuit, angles: list[float]) -> float:
        raise NotImplementedError
