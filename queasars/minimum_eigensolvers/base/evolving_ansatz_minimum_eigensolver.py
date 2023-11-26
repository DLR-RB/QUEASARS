# -*- coding: utf-8 -*-
# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Generic, TypeVar, Optional
from dataclasses import dataclass
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.list_or_dict import ListOrDict
from qiskit_algorithms.minimum_eigensolvers import (
    MinimumEigensolver,
    MinimumEigensolverResult,
)
from queasars.minimum_eigensolvers.base.bitstring_evaluator import BitstringEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import (
    Population,
    Operator,
    Individual,
    FitnessFunction,
)

POP = TypeVar("POP", bound=Population)
IND = TypeVar("IND", bound=Individual)
FIT = TypeVar("FIT", bound=FitnessFunction)


class EvolutionaryAlgorithm(Generic[POP, IND, FIT]):
    def __init__(
        self,
        initial_population: POP,
        fitness_function: FIT[IND],
        evolutionary_operators: list[Operator[POP[IND]]],
        n_generations: int,
    ):
        self.population: POP = initial_population
        self.fitness_function: FIT[IND] = fitness_function
        self.evolutionary_operators: list[Operator[POP[IND]]] = evolutionary_operators
        self.n_generations = n_generations

        for operator in evolutionary_operators:
            operator.initialize(fitness_function=fitness_function)

    def optimize(self) -> "EvolutionaryAlgorithmResult":
        for _ in range(self.n_generations):
            for operator in self.evolutionary_operators:
                self.population = operator.apply_operator(population=self.population)
            # check for termination conditions

        result: EvolutionaryAlgorithmResult[POP, IND] = EvolutionaryAlgorithmResult(
            final_population=self.population, best_individual=None
        )


@dataclass
class EvolutionaryAlgorithmResult(Generic[POP, IND]):
    final_population: POP[IND]
    best_individual: IND


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
