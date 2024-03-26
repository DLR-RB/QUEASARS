# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from abc import ABC, abstractmethod
from math import ceil
from typing import Callable, Optional

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BaseIndividual, BasePopulationEvaluationResult


class EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion(ABC):
    """Abstract class for holding and determining the termination state of an EvolvingAnsatzMinimumEigensolver"""

    @abstractmethod
    def reset_state(self) -> None:
        """Resets the internal state of the termination criteria to allow it's reuse in a new optimization run"""

    @abstractmethod
    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:
        """Checks for the latest evaluated population, whether the EvolvingAnsatzMinimumEigensolver should terminate

        :arg population_evaluation: evaluation result of the latest generation
        :type population_evaluation:  BasePopulationEvaluationResult
        :arg best_individual: best individual found so far by the solver, might not be part of the latest generation
        :type best_individual: BaseIndividual
        :arg best_expectation_value: expectation value of the best individual found so far
        :type best_expectation_value: float

        :return: True if the algorithm should terminate, False otherwise
        :rtype: bool
        """


class BestIndividualChangeTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):
    """Termination criterion which terminates when the absolute change in expectation value between the best individual
    of the current and the last generation falls below a threshold value

    :param minimum_change: absolute expectation value improvement below which the algorithm shall terminate.
        Must be at least 0
    :type minimum_change: float
    """

    def __init__(self, minimum_change: float):
        if minimum_change <= 0:
            raise ValueError("The minimum absolute improvement parameter must be bigger than 0!")

        self._minimum_change: float = minimum_change
        self._previous_expectation_value: float = float("inf")

    def reset_state(self) -> None:
        self._previous_expectation_value = float("inf")

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:
        if self._previous_expectation_value == float("inf"):
            self._previous_expectation_value = population_evaluation.best_expectation_value
            return False

        change: float = abs(self._previous_expectation_value - population_evaluation.best_expectation_value)
        if change >= self._minimum_change:
            self._previous_expectation_value = population_evaluation.best_expectation_value
            return False

        return True


class BestIndividualRelativeChangeTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):
    """Termination criterion which terminates when the absolute change in expectation value between
    the best individual of the current and previous generation falls below a threshold. This threshold is taken
    as a percentage of the expectation value of the best individual of the last generation

    :param minimum_relative_change: relative improvement in expectation value below which the algorithm shall
        terminate. Must be in the range )0,1)
    :type minimum_relative_change: float
    """

    def __init__(self, minimum_relative_change: float):
        if minimum_relative_change <= 0 or minimum_relative_change > 1:
            raise ValueError("The minimum relative improvement parameter must not exceed the range )0,1)!")

        self._minimum_relative_change = minimum_relative_change
        self._previous_expectation_value: float = float("inf")

    def reset_state(self) -> None:
        self._previous_expectation_value = float("inf")

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:
        if self._previous_expectation_value == float("inf"):
            self._previous_expectation_value = population_evaluation.best_expectation_value
            return False

        relative_change = abs(self._previous_expectation_value - population_evaluation.best_expectation_value) / abs(
            self._previous_expectation_value
        )
        if relative_change >= self._minimum_relative_change:
            self._previous_expectation_value = population_evaluation.best_expectation_value
            return False

        return True


class BestIndividualAbsoluteExpectationValueTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):

    def __init__(self, expectation_threshold: float):
        self._expectation_threshold: float = expectation_threshold

    def reset_state(self) -> None:
        pass

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:
        if best_expectation_value < self._expectation_threshold:
            return True
        return False


class AverageHausdorffDistanceTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):

    def __init__(
        self,
        distance_measure: Callable[[tuple[BaseIndividual, float], tuple[BaseIndividual, float]], float],
        distance_threshold: float,
        quantile: float,
    ):
        self._distance_measure: Callable[[tuple[BaseIndividual, float], tuple[BaseIndividual, float]], float] = (
            distance_measure
        )
        self._distance_threshold: float = distance_threshold
        self._quantile: float = quantile
        self._last_individuals: Optional[tuple[tuple[BaseIndividual, float], ...]] = None

    def _generational_distance(
        self,
        from_population: tuple[tuple[BaseIndividual, float], ...],
        to_population: tuple[tuple[BaseIndividual, float], ...],
    ) -> float:
        distance_sum: float = 0
        for from_individual in from_population:
            distance_sum += min(
                self._distance_measure(from_individual, to_individual) for to_individual in to_population
            )
        return distance_sum / len(from_population)

    def reset_state(self) -> None:
        pass

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:

        results: tuple[tuple[BaseIndividual, Optional[float]], ...] = tuple(
            zip(population_evaluation.population.individuals, population_evaluation.expectation_values)
        )
        filtered_results: tuple[tuple[BaseIndividual, float], ...] = tuple(
            (ind, fit) for ind, fit in results if fit is not None
        )
        filtered_results = tuple(sorted(filtered_results, key=lambda x: x[1]))
        quantile_index: int = ceil(len(filtered_results) * self._quantile)
        individuals: tuple[tuple[BaseIndividual, float], ...] = tuple(
            individual for individual in filtered_results[:quantile_index]
        )

        if self._last_individuals is not None:
            average_hausdorff_distance = max(
                self._generational_distance(self._last_individuals, individuals),
                self._generational_distance(individuals, self._last_individuals),
            )

            print("distance:", average_hausdorff_distance)

            if average_hausdorff_distance < self._distance_threshold:
                return True

        self._last_individuals = individuals
        return False
