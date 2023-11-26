# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.minimum_eigensolvers import (
    MinimumEigensolver,
    MinimumEigensolverResult,
)

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator


class EvolvingAnsatzMinimumEigensolver(MinimumEigensolver):
    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        pass

    def compute_minimum_function_value(
        self,
        operator: BitstringEvaluator,
        aux_operators: ListOrDict[BitstringEvaluator] | None = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        pass


class EvolvingAnsatzMinimumEigensolverResult(MinimumEigensolverResult):
    pass
