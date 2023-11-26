# -*- coding: utf-8 -*-
# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.list_or_dict import ListOrDict

from queasars.minimum_eigensolvers.base.bitstring_evaluator import BitstringEvaluator
from queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver import (
    EvolvingAnsatzMinimumEigensolver,
    EvolvingAnsatzMinimumEigensolverResult,
)


@dataclass
class EVQEConfiguration:
    """"""

    pass


class EVQE(EvolvingAnsatzMinimumEigensolver):

    configuration: EVQEConfiguration

    def __init__(self, configuration: EVQEConfiguration):
        super().__init__()
        self.configuration = configuration

    def compute_minimum_eigenvalue(
        self,
        operator: BaseOperator,
        aux_operators: ListOrDict[BaseOperator] | None = None,
    ) -> EvolvingAnsatzMinimumEigensolverResult:
        pass

    def compute_minimum_function_value(
        self,
        operator: BitstringEvaluator,
        aux_operators: ListOrDict[BitstringEvaluator, BaseOperator] | None = None,
    ) -> "EvolvingAnsatzMinimumEigensolverResult":
        pass

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True
