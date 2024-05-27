# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from concurrent.futures import Future as ConcurrentFuture, wait as concurrent_wait
from random import Random
from typing import Optional, Union, cast
from warnings import warn

from dask.distributed import Future as DaskFuture, wait as dask_wait, Client
from numpy import argmin

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import (
    BaseEvolutionaryOperator,
    OperatorContext,
    BasePopulationEvaluationResult,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import EVQEIndividual
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.population import EVQEPopulation


class EVQESelection(BaseEvolutionaryOperator[EVQEPopulation]):
    """
    Class representing the selection evolutionary operator of the EVQE algorithm. Directly before this
    Operator is applied a speciation Operator must have been applied. In other words, the EVQEPopulation's
    species_representatives, species_members and species_membership attributes may not be None when this Operator
    is applied

    :param alpha_penalty: scaling factor for penalizing the amount of circuit layers of an individual
    :type: float
    :param beta_penalty: scaling factor for penalizing the amount of controlled gates of an individual
    :type: float
    :param use_tournament_selection: indicates whether to use tournament selection. By default, this is
        set to False. In that case, roulette wheel selection is used. Should be true, if the measured expectation
        values can be negative.
    :type use_tournament_selection: bool
    :param tournament_size: indicates the size of the tournaments used. This can be in the range [1, population_size].
        It cannot be None, if use_tournament_selection is set to True. A tournament_size of 1 yields random selection,
        with increasing tournament selection sizes increasing the selection pressure.
    :type tournament_size: int
    :param random_seed: integer value to control randomness
    :type: int
    """

    def __init__(
        self,
        alpha_penalty: float,
        beta_penalty: float,
        use_tournament_selection: bool = False,
        tournament_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        self._alpha_penalty: float = alpha_penalty
        self._beta_penalty: float = beta_penalty
        self._use_tournament_selection: bool = use_tournament_selection
        if self._use_tournament_selection:
            if tournament_size is not None:
                self._tournament_size: int = tournament_size
                if self._tournament_size < 1:
                    raise ValueError("the tournament_size must be at least 1!")
            else:
                raise ValueError("tournament_size cannot be None, if tournament selection should be used!")
        self._random_generator: Random = Random(random_seed)

    def apply_operator(self, population: EVQEPopulation, operator_context: OperatorContext) -> EVQEPopulation:
        # measure the expectation values for all individuals

        future_evaluation_results: list[Union[DaskFuture, ConcurrentFuture]] = []
        if isinstance(operator_context.parallel_executor, Client):
            cast(list[DaskFuture], future_evaluation_results)
            wait = dask_wait
        else:
            cast(list[ConcurrentFuture], future_evaluation_results)
            wait = concurrent_wait

        future_evaluation_results = [
            operator_context.parallel_executor.submit(
                operator_context.circuit_evaluator.evaluate_circuits,
                [individual.get_parameterized_quantum_circuit()],
                [list(individual.get_parameter_values())],
            )
            for individual in population.individuals
        ]

        wait(future_evaluation_results)
        evaluation_results: list[float] = [future.result()[0] for future in future_evaluation_results]
        del future_evaluation_results
        operator_context.circuit_evaluation_count_callback(len(population.individuals))

        # cannot apply selection if speciation was not done beforehand
        if (
            population.species_representatives is None
            or population.species_members is None
            or population.species_membership is None
        ):
            raise EVQESelectionException(
                "Selection can't be finished if speciation information is missing!\n"
                + "Either the species_representatives, species_members or species_membership\n"
                + "attribute of the population is None!"
            )

        # report the result of the evaluation
        best_individual_index: int = int(argmin(evaluation_results))
        result: BasePopulationEvaluationResult[EVQEIndividual] = BasePopulationEvaluationResult(
            population=population,
            expectation_values=tuple(evaluation_results),
            best_individual=population.individuals[best_individual_index],
            best_expectation_value=evaluation_results[best_individual_index],
        )
        operator_context.result_callback(result)

        selected_individuals: list[EVQEIndividual] = []
        fitness_values: list[float] = []

        if not self._use_tournament_selection:
            # disallow negative or 0 values in fitnesses by shifting all evaluation results by a fixed offset
            offset: float
            if evaluation_results[best_individual_index] <= 0:
                offset = -evaluation_results[best_individual_index] + 1
                warn(
                    "Tournament selection should be preferred over roulette wheel selection, "
                    + "if negative expectation values are involved in the fitness!"
                )
            else:
                offset = 0

            fitness_values = [
                (
                    evaluation_results[i]
                    + offset
                    + self._alpha_penalty * len(individual.layers)
                    + self._beta_penalty * individual.get_n_controlled_gates()
                )
                * float(len(population.species_members[population.species_membership[i]]))
                for i, individual in enumerate(population.individuals)
            ]

            fitness_weights: list[float] = [1 / (fitness + offset) for fitness in fitness_values]
            selected_individuals = self._random_generator.choices(
                population.individuals, weights=fitness_weights, k=len(population.individuals)
            )
        else:

            fitness_values = [
                (
                    evaluation_results[i]
                    + self._alpha_penalty * len(individual.layers)
                    + self._beta_penalty * individual.get_n_controlled_gates()
                )
                * float(len(population.species_members[population.species_membership[i]]))
                for i, individual in enumerate(population.individuals)
            ]

            while len(selected_individuals) < len(population.individuals):
                tournament_indices = self._random_generator.choices(
                    range(0, len(population.individuals)), k=self._tournament_size
                )
                best_index: Optional[int] = None
                best_fitness_value: Optional[float] = None

                for tournament_index in tournament_indices:
                    if best_fitness_value is None or fitness_values[tournament_index] < best_fitness_value:
                        best_index = tournament_index
                        best_fitness_value = fitness_values[tournament_index]

                if best_index is not None:
                    selected_individuals.append(population.individuals[best_index])
                else:
                    raise EVQESelectionException("No individual was selected for a round of tournament selection!")

        return EVQEPopulation(
            individuals=tuple(selected_individuals),
            species_representatives=population.species_representatives,
            species_members=None,
            species_membership=None,
        )

    def get_n_expected_circuit_evaluations(
        self, population: EVQEPopulation, operator_context: OperatorContext
    ) -> Optional[int]:
        return len(population.individuals)


class EVQESelectionException(Exception):
    """Class to represent errors caused during the EVQE selection operator"""
