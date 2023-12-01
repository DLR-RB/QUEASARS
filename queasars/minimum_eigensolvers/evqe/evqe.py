# -*- coding: utf-8 -*-
# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.list_or_dict import ListOrDict

from queasars.circuit_evaluation.bitstring_evaluation import BitstringEvaluator
from queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver import (
    EvolvingAnsatzMinimumEigensolver,
    EvolvingAnsatzMinimumEigensolverResult,
)


@dataclass
class EVQEMinimumEigensolverConfiguration:
    """Configuration for the EVQEMinimumEigensolver"""


class EVQEMinimumEigensolver(EvolvingAnsatzMinimumEigensolver):
    """Minimum eigensolver which uses the `EVQE` algorithm. For details see: https://arxiv.org/abs/1910.09694"""

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> EvolvingAnsatzMinimumEigensolverResult:
        return super().compute_minimum_eigenvalue(operator, aux_operators)

    def compute_minimum_function_value(
        self,
        operator: BitstringEvaluator,
        aux_operators: ListOrDict[BitstringEvaluator, BaseOperator] | None = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        return super().compute_minimum_function_value(operator, aux_operators)

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True
