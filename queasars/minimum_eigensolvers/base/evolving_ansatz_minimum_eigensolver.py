# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from typing import Any

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolver
from qiskit_algorithms.algorithm_result import AlgorithmResult
from qiskit.circuit import QuantumCircuit
from qiskit.result import QuasiDistribution

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import Population


class EvolvingAnsatzMinimumEigensolver(ABC, MinimumEigensolver):
    """"""

    @abstractmethod
    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        """"""

    @abstractmethod
    def compute_minimum_function_value(
        self,
        operator: BitstringEvaluator,
        aux_operators: ListOrDict[BitstringEvaluator] | None = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        """"""


class EvolvingAnsatzMinimumEigensolverResult(AlgorithmResult):
    """"""

    def __init__(self) -> None:
        super().__init__()
        self._eigenvalue: complex | None = None
        self._eigenstate: QuasiDistribution | None = None
        self._aux_operator_values: ListOrDict[tuple[complex, dict[str, Any]]] | None = None
        self._optimal_parameters: dict | None = None
        self._optimal_circuit: QuantumCircuit | None = None
        self._cost_function_evals: int | None = None
        self._generations: int | None = None
        self._final_population: Population | None = None
        self._best_measurement: dict[str, Any] | None = None

    @property
    def eigenvalue(self) -> complex | None:
        """"""
        return self._eigenvalue

    @eigenvalue.setter
    def eigenvalue(self, value: complex):
        """"""
        self._eigenvalue = value

    @property
    def eigenstate(self) -> QuasiDistribution | None:
        """"""
        return self._eigenstate

    @eigenstate.setter
    def eigenstate(self, value: QuasiDistribution):
        """"""
        self._eigenstate = value

    @property
    def aux_operators_evaluated(
        self,
    ) -> ListOrDict[tuple[complex, dict[str, Any]]] | None:
        """Return aux operator expectation values and metadata.

        These are formatted as (mean, metadata).
        """
        return self._aux_operator_values

    @aux_operators_evaluated.setter
    def aux_operators_evaluated(self, value: ListOrDict[tuple[complex, dict[str, Any]]] | None) -> None:
        """"""
        self._aux_operator_values = value

    @property
    def optimal_parameters(self) -> dict | None:
        """Returns the optimal parameters in a dictionary"""
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: dict) -> None:
        """Sets optimal parameters"""
        self._optimal_parameters = value

    @property
    def optimal_circuit(self) -> QuantumCircuit:
        """The optimal circuits. Along with the optimal parameters,
        these can be used to retrieve the minimum eigenstate.
        """
        return self._optimal_circuit

    @optimal_circuit.setter
    def optimal_circuit(self, optimal_circuit: QuantumCircuit) -> None:
        """"""
        self._optimal_circuit = optimal_circuit

    @property
    def cost_function_evals(self) -> int | None:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def generations(self) -> int | None:
        """"""
        return self._generations

    @generations.setter
    def generations(self, value: int):
        """"""
        self._generations = value

    @property
    def final_population(self) -> Population | None:
        """"""
        return self._final_population

    @final_population.setter
    def final_population(self, value: Population):
        """"""
        self._final_population = value

    @property
    def best_measurement(self) -> dict[str, Any] | None:
        """Return the best measurement over the entire optimization.

        Possesses keys: ``state``, ``bitstring``, ``value``, ``probability``.
        """
        return self._best_measurement

    @best_measurement.setter
    def best_measurement(self, value: dict[str, Any]) -> None:
        """Set the best measurement over the entire optimization."""
        self._best_measurement = value
