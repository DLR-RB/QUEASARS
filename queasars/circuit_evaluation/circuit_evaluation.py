# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from typing import Union, Optional
from warnings import warn

from numpy import real
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseEstimator, BaseSampler, EstimatorResult, SamplerResult, PrimitiveJob
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.result import QuasiDistribution

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator
from queasars.circuit_evaluation.expectation_calculation import (
    get_expectation_with_operator,
    get_expectation_with_bitstring_evaluator,
)


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
    """Class which evaluates quantum circuits by estimating the expectation value of the circuit for a given operator

    :param qiskit_primitive: Qiskit primitive used for estimating the circuit's expectation value.
        Must be an Estimator or a Sampler. The usage of an Estimator is the preferred option
    :type qiskit_primitive: BaseEstimator | BaseSampler
    :param operator: Operator for which the expectation value is estimated. If the operator is not hermitian,
        the complex part of the result is dropped. If the qiskit_primitive is a sampler, the operator
        must be a diagonal SparsePauliOp
    :type operator: BaseOperator
    :param alpha: when using a Sampler to estimate the expectation value, it is derived from the probability
        distribution of measured basis states. In that case, the expectation value can also be calculated
        over only the lower alpha tail of the state distribution. Alpha must be in the range (0, 1].
        By default, alpha is 1, recovering the normal expectation value calculation. For other alpha values
        this leads to the CVaR score as explained in https://quantum-journal.org/papers/q-2020-04-20-256/ .
        If using the CVaR score, the operator must be a SparsePauliOp
    :type alpha: float
    :param initial_state_circuit: If the qubits of the quantum circuits shall be initialized in a specific
        state for the circuit evaluation, this initial state can be given as the quantum circuit which initializes
        it. It must operate on exactly as many qubits as the given operator
    :type initial_state_circuit: Optional[QuantumCircuit]
    """

    def __init__(
        self,
        qiskit_primitive: Union[BaseEstimator[PrimitiveJob[EstimatorResult]], BaseSampler[PrimitiveJob[SamplerResult]]],
        operator: BaseOperator,
        alpha: float,
        initial_state_circuit: Optional[QuantumCircuit] = None,
    ):
        """Constructor Method"""
        self._estimator: Optional[BaseEstimator[PrimitiveJob[EstimatorResult]]] = None
        self._sampler: Optional[BaseSampler[PrimitiveJob[SamplerResult]]] = None
        if isinstance(qiskit_primitive, BaseEstimator):
            self._estimator = qiskit_primitive
        elif isinstance(qiskit_primitive, BaseSampler):
            self._sampler = qiskit_primitive
        else:
            raise ValueError("The qiskit primitive was neither a BaseEstimator nor a BaseSampler!")

        if self._sampler is not None and not isinstance(operator, SparsePauliOp):
            raise ValueError(
                "If using a sampler to estimate the expectation value, the operator must be a SparsePauliOp!"
            )
        self._operator: BaseOperator = operator

        if alpha <= 0 or 1 < alpha:
            raise ValueError("alpha must be in the range (0, 1]!")
        if alpha < 1 and self._estimator is not None:
            warn(
                "If an estimator is used to calculate the expectation value, specifying the alpha value has no effect!"
            )
        self._alpha: float = alpha
        if initial_state_circuit is not None and initial_state_circuit.num_qubits != self._operator.num_qubits:
            raise ValueError(
                f"The amount of qubits in the initial state circuit ({initial_state_circuit.num_qubits} "
                + f"does not match the amount of qubits in the given operator ({self._operator.num_qubits})"
            )
        self._initial_state_circuit: Optional[QuantumCircuit] = initial_state_circuit

    def evaluate_circuits(self, circuits: list[QuantumCircuit], parameter_values: list[list[float]]) -> list[float]:
        if self._initial_state_circuit is not None:
            circuits = [self._initial_state_circuit.compose(circuit, inplace=False) for circuit in circuits]

        if self._sampler is not None and isinstance(self._operator, SparsePauliOp):
            circuits = [circuit.measure_all(inplace=False) for circuit in circuits]
            sampling_result: SamplerResult = self._sampler.run(
                circuits=circuits, parameter_values=parameter_values
            ).result()
            quasi_dists: list[QuasiDistribution] = sampling_result.quasi_dists
            return [
                get_expectation_with_operator(measurement_distribution=dist, operator=self._operator, alpha=self._alpha)
                for dist in quasi_dists
            ]

        if self._estimator is not None:
            evaluation_result: EstimatorResult = self._estimator.run(
                circuits=circuits, observables=[self._operator] * len(circuits), parameter_values=parameter_values
            ).result()
            return list(real(evaluation_result.values))

        raise ValueError(
            "The OperatorCircuitEvaluator was unable to return results, as it seems to have been misconfigured!"
        )

    @property
    def n_qubits(self) -> int:
        return self._operator.num_qubits


class BitstringCircuitEvaluator(BaseCircuitEvaluator):
    """Class which evaluates quantum circuits by calculating an expectation value over the circuit's sampled
    measurement distribution using the :class:`queasars.circuit_evaluation.bitstring_evaluation.BitstringEvaluator`
    to assign floating point values to individual measurements

    :param sampler: Sampler primitive to retrieve measurement distributions with
    :type sampler: BaseSampler
    :param bitstring_evaluator: Evaluation function used to evaluate individual measurements
    :type bitstring_evaluator: BitstringEvaluator
    :param alpha: when using a Sampler to estimate the expectation value, it is derived from the probability
        distribution of measured basis states. In that case, the expectation value can also be calculated
        over only the lower alpha tail of the state distribution. Alpha must be in the range (0, 1].
        By default, alpha is 1, recovering the normal expectation value calculation. For other alpha values
        this leads to the CVaR score as explained in https://quantum-journal.org/papers/q-2020-04-20-256/ .
        If using the CVaR score, the operator must be a SparsePauliOp
    :type alpha: float
    :param initial_state_circuit: If the qubits of the quantum circuits shall be initialized in a specific
        state for the circuit evaluation, this initial state can be given as the quantum circuit which initializes
        it. It must operate on exactly as many qubits as the input length of the bitstring_evaluator specifies
    :type initial_state_circuit: Optional[QuantumCircuit]
    """

    def __init__(
        self,
        sampler: BaseSampler[PrimitiveJob[SamplerResult]],
        bitstring_evaluator: BitstringEvaluator,
        alpha: float,
        initial_state_circuit: Optional[QuantumCircuit] = None,
    ):
        """Constructor Method"""
        self._sampler: BaseSampler[PrimitiveJob[SamplerResult]] = sampler
        self._bitstring_evaluator: BitstringEvaluator = bitstring_evaluator
        if (
            initial_state_circuit is not None
            and initial_state_circuit.num_qubits != self._bitstring_evaluator.input_length
        ):
            raise ValueError(
                f"The amount of qubits in the initial state circuit ({initial_state_circuit.num_qubits} "
                + "does not match the input length of the BitstringEvaluator "
                + f"({self._bitstring_evaluator.input_length})!"
            )
        if alpha <= 0 or 1 < alpha:
            raise ValueError("alpha must be in the range (0, 1]!")
        self._alpha: float = alpha
        self._initial_state_circuit: Optional[QuantumCircuit] = initial_state_circuit

    def evaluate_circuits(self, circuits: list[QuantumCircuit], parameter_values: list[list[float]]) -> list[float]:
        if self._initial_state_circuit is not None:
            circuits = [self._initial_state_circuit.compose(circuit, inplace=False) for circuit in circuits]

        circuits = [circuit.measure_all(inplace=False) for circuit in circuits]
        evaluation_result: SamplerResult = self._sampler.run(
            circuits=circuits, parameter_values=parameter_values
        ).result()
        quasi_dists: list[QuasiDistribution] = evaluation_result.quasi_dists

        return [
            get_expectation_with_bitstring_evaluator(
                measurement_distribution=dist, bitstring_evaluator=self._bitstring_evaluator, alpha=self._alpha
            )
            for dist in quasi_dists
        ]

    @property
    def n_qubits(self) -> int:
        return self._bitstring_evaluator.input_length
