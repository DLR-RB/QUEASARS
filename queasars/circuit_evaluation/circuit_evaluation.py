# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from typing import Optional

from numpy import real
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import (
    BaseEstimatorV2,
    EstimatorPubLike,
    SamplerPubLike,
    BaseSamplerV2,
    PrimitiveResult,
    PubResult,
    SamplerPubResult,
)
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.result import QuasiDistribution

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator
from queasars.circuit_evaluation.expectation_calculation import (
    get_expectation_with_operator,
    get_expectation_with_bitstring_evaluator,
)


def measure_quasi_distributions(
    circuits: list[QuantumCircuit], parameter_values: list[list[float]], sampler: BaseSamplerV2, shots: int
) -> list[QuasiDistribution]:
    """
    In qiskit's move from SamplerV1 to SamplerV2 primitives, the sampler primitives no longer provide
    a QuasiDistribution but yield shot counts instead. Given parameterized quantum circuits, parameter values,
    a SamplerV2 primitive and a shot count, this method wraps the sampler call to calculate the QuasiDistributions
    based on the shot counts for each quantum circuit.

    :arg circuits: Parameterized quantum circuits to measure QuasiDistributions from.
    :type circuits: list[QuantumCircuit]
    :arg parameter_values: Parameter values to populate the parameterized quantum circuits with.
    :type parameter_values: list[list[float]]
    :arg sampler: Qiskit SamplerV2 primitive to measure shot counts with.
    :type sampler: BaseSamplerV2
    :arg shots: Amount of measurements ("shots") per quantum circuit to approximate a QuasiDistribution with.
    :type shots: int

    :return: A list of QuasiDistributions
    :rtype: list[QuasiDistribution]
    """
    circuits = [circuit.measure_all(inplace=False) for circuit in circuits]
    pubs: tuple[SamplerPubLike, ...] = tuple(
        (circ, params) for circ, params in zip(circuits, parameter_values) if circ is not None and params is not None
    )
    result: PrimitiveResult[SamplerPubResult] = sampler.run(pubs=pubs, shots=shots).result()
    counts: list[dict[str, float]] = [res.data["meas"].get_counts() for res in result]
    return [
        QuasiDistribution(data={state: count / shots for state, count in count_dict.items()}, shots=shots)
        for count_dict in counts
    ]


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


class OperatorSamplerCircuitEvaluator(BaseCircuitEvaluator):
    """
    Class which evaluates quantum circuits by approximating the expectation value of the circuit
    for a given operator using the qiskit SamplerV2 primitive

    :param sampler: Qiskit SamplerV2 primitive used to approximate the expectation value
    :type sampler: BaseSamplerV2
    :param sampler_shots: Amount of circuit measurements ("shots") used to approximate the expectation value
    :type sampler_shots: int
    :param operator: Operator for which the expectation value is estimated
    :type operator: SparsePauliOp
    :param alpha: The expectation value is approximated based on the probability distribution of measured basis states
        and their respective eigenvalues. The expectation value can also be calculated over only the lower alpha tail
        of the state distribution. Alpha must be in the range (0, 1]. By default, alpha is 1, recovering the normal
        expectation value calculation. For other alpha values this leads to the CVaR score as explained in
        https://quantum-journal.org/papers/q-2020-04-20-256/.
    :type alpha: float
    :param initial_state_circuit: If the qubits of the quantum circuits shall be initialized in a specific
        state for the circuit evaluation, this initial state can be given as the quantum circuit which initializes
        it. It must operate on exactly as many qubits as the given operator
    :type initial_state_circuit: Optional[QuantumCircuit]
    """

    def __init__(
        self,
        sampler: BaseSamplerV2,
        sampler_shots: int,
        operator: SparsePauliOp,
        alpha: float = 1.0,
        initial_state_circuit: Optional[QuantumCircuit] = None,
    ):
        """Constructor Method"""
        self._sampler: BaseSamplerV2 = sampler

        if not isinstance(operator, SparsePauliOp):
            raise ValueError(
                "If using a sampler to estimate the expectation value, the operator must be a SparsePauliOp!"
            )
        self._operator: SparsePauliOp = operator

        self._sampler_shots: int = sampler_shots

        if alpha <= 0 or 1 < alpha:
            raise ValueError("alpha must be in the range (0, 1]!")
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

        quasi_dists: list[QuasiDistribution] = measure_quasi_distributions(
            circuits=circuits, parameter_values=parameter_values, sampler=self._sampler, shots=self._sampler_shots
        )
        return [
            get_expectation_with_operator(measurement_distribution=dist, operator=self._operator, alpha=self._alpha)
            for dist in quasi_dists
        ]

    @property
    def n_qubits(self) -> int:
        return self._operator.num_qubits


class OperatorCircuitEvaluator(BaseCircuitEvaluator):
    """Class which evaluates quantum circuits by approximating the expectation value of the circuit
    for a given operator using the qiskit EstimatorV2 primitive

    :param estimator: Qiskit EstimatorV2 primitive used to approximate the expectation value
    :type estimator: BaseEstimatorV2
    :param estimator_precision: Precision to which the expectation value is approximated
    :type estimator_precision: float
    :param operator: Operator for which the expectation value is estimated. If the operator is not hermitian,
        the complex part of the result is dropped
    :type operator: BaseOperator
    :param initial_state_circuit: If the qubits of the quantum circuits shall be initialized in a specific
        state for the circuit evaluation, this initial state can be given as the quantum circuit which initializes
        it. It must operate on exactly as many qubits as the given operator
    :type initial_state_circuit: Optional[QuantumCircuit]
    """

    def __init__(
        self,
        estimator: BaseEstimatorV2,
        estimator_precision: float,
        operator: BaseOperator,
        initial_state_circuit: Optional[QuantumCircuit] = None,
    ):
        """Constructor Method"""
        self._estimator: BaseEstimatorV2 = estimator
        self._estimator_precision: float = estimator_precision
        self._operator: BaseOperator = operator

        if initial_state_circuit is not None and initial_state_circuit.num_qubits != self._operator.num_qubits:
            raise ValueError(
                f"The amount of qubits in the initial state circuit ({initial_state_circuit.num_qubits} "
                + f"does not match the amount of qubits in the given operator ({self._operator.num_qubits})"
            )
        self._initial_state_circuit: Optional[QuantumCircuit] = initial_state_circuit

    def evaluate_circuits(self, circuits: list[QuantumCircuit], parameter_values: list[list[float]]) -> list[float]:
        if self._initial_state_circuit is not None:
            circuits = [self._initial_state_circuit.compose(circuit, inplace=False) for circuit in circuits]

        pubs: tuple[EstimatorPubLike, ...] = tuple(
            (circ, self._operator, params)
            for circ, params in zip(circuits, parameter_values)
            if circ is not None and params is not None
        )

        result: PrimitiveResult[PubResult] = self._estimator.run(
            pubs=pubs,
            precision=self._estimator_precision,
        ).result()

        return [real(res.data.evs) for res in result]

    @property
    def n_qubits(self) -> int:
        return self._operator.num_qubits


class BitstringCircuitEvaluator(BaseCircuitEvaluator):
    """Class which evaluates quantum circuits by approximating an expectation value over the circuit's sampled
    measurement distribution using the :class:`queasars.circuit_evaluation.bitstring_evaluation.BitstringEvaluator`
    to assign floating point values to individual measurements

    :param sampler: Qiskit SamplerV2 primitive to retrieve measurement distributions with
    :type sampler: BaseSamplerV2
    :param sampler_shots: Amount of circuit measurements ("shots") used to approximate the expectation value
    :type sampler_shots: int
    :param bitstring_evaluator: Evaluation function used to evaluate individual measurements
    :type bitstring_evaluator: BitstringEvaluator
    :param alpha: The expectation value is approximated based on the probability distribution of measured basis states
        and their respective bitstring evaluation values. The expectation value can also be calculated over only the
        lower alpha tail of the state distribution. Alpha must be in the range (0, 1]. By default, alpha is 1,
        recovering the normal expectation value calculation. For other alpha values this leads to the CVaR score
        as explained in https://quantum-journal.org/papers/q-2020-04-20-256/.
    :type alpha: float
    :param initial_state_circuit: If the qubits of the quantum circuits shall be initialized in a specific
        state for the circuit evaluation, this initial state can be given as the quantum circuit which initializes
        it. It must operate on exactly as many qubits as the input length of the bitstring_evaluator specifies
    :type initial_state_circuit: Optional[QuantumCircuit]
    """

    def __init__(
        self,
        sampler: BaseSamplerV2,
        sampler_shots: int,
        bitstring_evaluator: BitstringEvaluator,
        alpha: float = 1.0,
        initial_state_circuit: Optional[QuantumCircuit] = None,
    ):
        """Constructor Method"""
        self._sampler: BaseSamplerV2 = sampler
        self._sampler_shots: int = sampler_shots
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

        quasi_dists: list[QuasiDistribution] = measure_quasi_distributions(
            circuits=circuits,
            parameter_values=parameter_values,
            sampler=self._sampler,
            shots=self._sampler_shots,
        )

        return [
            get_expectation_with_bitstring_evaluator(
                measurement_distribution=dist, bitstring_evaluator=self._bitstring_evaluator, alpha=self._alpha
            )
            for dist in quasi_dists
        ]

    @property
    def n_qubits(self) -> int:
        return self._bitstring_evaluator.input_length
