# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from abc import ABC, abstractmethod
from typing import Optional, cast
from numpy import median

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
    :param allowed_consecutive_violations: determines how often the threshold value can be violated consecutively
        before this criterion chooses to terminate. If set to 0, this terminates the first time the change falls
        below the threshold value. If set to 2 for example, this terminates the first time the change falls below
        the threshold three consecutive times. Must be at least 0.
    :type allowed_consecutive_violations: int
    """

    def __init__(self, minimum_change: float, allowed_consecutive_violations: int = 0):
        if minimum_change <= 0:
            raise ValueError("The minimum absolute improvement parameter must be bigger than 0!")
        if allowed_consecutive_violations < 0:
            raise ValueError("allowed_consecutive_violations must be at least 0!")

        self._minimum_change: float = minimum_change
        self._allowed_consecutive_violations: int = allowed_consecutive_violations
        self._previous_expectation_value: Optional[float] = None
        self._change_history: list[float] = []

    def reset_state(self) -> None:
        self._previous_expectation_value = None
        self._change_history = []

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:

        if self._previous_expectation_value is None:
            self._previous_expectation_value = population_evaluation.best_expectation_value
            return False

        change = abs(self._previous_expectation_value - population_evaluation.best_expectation_value)
        self._change_history.append(change)
        self._previous_expectation_value = population_evaluation.best_expectation_value

        if len(self._change_history) < self._allowed_consecutive_violations + 1:
            return False

        max_change_in_last_relevant_window = max(self._change_history[-self._allowed_consecutive_violations - 1 :])

        return max_change_in_last_relevant_window < self._minimum_change


class BestIndividualRelativeChangeTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):
    """Termination criterion which terminates when the absolute change in expectation value between
    the best individual of the current and previous generation falls below a threshold. This threshold is taken
    as a percentage of the expectation value of the best individual of the last generation

    :param minimum_relative_change: relative improvement in expectation value below which the algorithm shall
        terminate. Must be in the range )0,1)
    :type minimum_relative_change: float
    :param allowed_consecutive_violations: determines how often the threshold value can be violated consecutively
        before this criterion chooses to terminate. If set to 0, this terminates the first time the change falls
        below the threshold value. If set to 2 for example, this terminates the first time the change falls below
        the threshold three consecutive times. Must be at least 0.
    :type allowed_consecutive_violations: int
    """

    def __init__(self, minimum_relative_change: float, allowed_consecutive_violations: int = 0):
        if minimum_relative_change <= 0 or minimum_relative_change > 1:
            raise ValueError("The minimum relative improvement parameter must not exceed the range )0,1)!")
        if allowed_consecutive_violations < 0:
            raise ValueError("allowed_consecutive_violations must be at least 0!")

        self._minimum_relative_change: float = minimum_relative_change
        self._allowed_consecutive_violations: int = allowed_consecutive_violations
        self._previous_expectation_value: Optional[float] = None
        self._relative_change_history: list[float] = []

    def reset_state(self) -> None:
        self._previous_expectation_value = None
        self._relative_change_history = []

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:

        if self._previous_expectation_value is None:
            self._previous_expectation_value = population_evaluation.best_expectation_value
            return False

        relative_change = abs(self._previous_expectation_value - population_evaluation.best_expectation_value) / abs(
            self._previous_expectation_value
        )
        self._previous_expectation_value = population_evaluation.best_expectation_value
        self._relative_change_history.append(relative_change)

        if len(self._relative_change_history) < self._allowed_consecutive_violations + 1:
            return False

        max_change_in_last_relevant_window = max(
            self._relative_change_history[-self._allowed_consecutive_violations - 1 :]
        )

        return max_change_in_last_relevant_window < self._minimum_relative_change


class BestIndividualExpectationValueThreshold(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):
    """
    Termination criterion which terminates, if the expectation value of the best individual of a population
    falls below a certain threshold value

    :param expectation_threshold: expectation value below which the optimization terminates
    :type expectation_threshold: float
    """

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
        if population_evaluation.best_expectation_value < self._expectation_threshold:
            return True
        return False


def _median_hausdorff_distance_by_expectation_value(
    result_1: BasePopulationEvaluationResult, result_2: BasePopulationEvaluationResult
) -> float:
    """
    Calculates the median Hausdorff distance between the expectation values in two population evaluation results

    :arg result_1: first evaluation result to calculate the distance on
    :type result_1: BasePopulationEvaluationResult
    :arg result_2: second evaluation result to calculate the distance on
    :type result_2: BasePopulationEvaluationResult
    :return: the median Hausdorff distance between the expectation values in two population evaluation results
    """

    def distance(from_expectations: list[float], to_expectations: list[float]) -> float:
        distances: list[float] = []
        for from_expectation in from_expectations:
            distances.append(min(abs(from_expectation - to_expectation) for to_expectation in to_expectations))
        return cast(float, median(distances))

    expectations_1: list[float] = [exp for exp in result_1.expectation_values if exp is not None]
    expectations_2: list[float] = [exp for exp in result_2.expectation_values if exp is not None]
    return max(distance(expectations_1, expectations_2), distance(expectations_2, expectations_1))


class PopulationChangeTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):
    """
    Termination criterion which terminates, if the current generation has not changed sufficiently from
    the last generation. This change is taken as the maximum of the absolute change of the expectation value of
    the best individual from the last and current generation and the median Hausdorff distance between the
    expectation values of the last and current generation. This means this criterion terminates if the best individual
    AND the population as a whole do not change sufficiently. This termination criterion can be configured to only
    terminate if the change falls below the threshold value multiple times consecutively

    :param minimum_change: threshold value for the absolute change in expectation value, below which this criterion
        chooses to terminate the optimization
    :type minimum_change: float
    :param allowed_consecutive_violations: determines how often the threshold value can be violated consecutively
        before this criterion chooses to terminate. If set to 0, this terminates the first time the change falls
        below the threshold value. If set to 2 for example, this terminates the first time the change falls below
        the threshold three consecutive times. Must be at least 0.
    :type allowed_consecutive_violations: int
    """

    def __init__(self, minimum_change: float, allowed_consecutive_violations: int):
        super().__init__()
        self._minimum_change: float = minimum_change
        if allowed_consecutive_violations < 0:
            raise ValueError("allowed_consecutive_violations must be at least 0!")
        self._allowed_consecutive_violations: int = allowed_consecutive_violations
        self._change_history: list[float] = [
            10 * self._minimum_change for _ in range(0, self._allowed_consecutive_violations + 1)
        ]
        self._last_population_evaluation: Optional[BasePopulationEvaluationResult] = None

    def reset_state(self) -> None:
        self._change_history = [10 * self._minimum_change for _ in range(0, self._allowed_consecutive_violations + 1)]
        self._last_population_evaluation = None

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:

        if self._last_population_evaluation is not None:
            hausdorff_distance: float = _median_hausdorff_distance_by_expectation_value(
                self._last_population_evaluation, population_evaluation
            )
            best_individual_distance: float = abs(
                self._last_population_evaluation.best_expectation_value - population_evaluation.best_expectation_value
            )

            self._change_history.append(max(hausdorff_distance, best_individual_distance))

        self._last_population_evaluation = population_evaluation

        if len(self._change_history) < self._allowed_consecutive_violations + 1:
            return False

        if max(self._change_history[-(self._allowed_consecutive_violations + 1) :]) < self._minimum_change:
            return True

        return False


class PopulationChangeRelativeTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):
    """
    Termination criterion which terminates, if the current generation has not changed sufficiently from
    the last generation. The change is taken as the maximum of the absolute change of the expectation value of
    the best individual from the last and current generation and the median Hausdorff distance between the
    expectation values of the last and current generation. It is then set in relation to the median expectation
    value of the last generation. This means this criterion terminates if the best individual
    AND the population as a whole do not change sufficiently. This termination criterion can be configured to only
    terminate if the change falls below the threshold value multiple times consecutively

    :param minimum_relative_change: threshold value for the change in expectation value relative to the last
        generation's median expectation value, below which this criterion chooses to terminate the optimization
    :type minimum_relative_change: float
    :param allowed_consecutive_violations: determines how often the threshold value can be violated consecutively
        before this criterion chooses to terminate. If set to 0, this terminates the first time the change falls
        below the threshold value. If set to 2 for example, this terminates the first time the change falls below
        the threshold three consecutive times. Must be at least 0.
    :type allowed_consecutive_violations: int
    """

    def __init__(self, minimum_relative_change: float, allowed_consecutive_violations: int):
        super().__init__()
        self._minimum_relative_change: float = minimum_relative_change
        if allowed_consecutive_violations < 0:
            raise ValueError("allowed_consecutive_violations must be at least 0!")
        self._allowed_consecutive_violations: int = allowed_consecutive_violations
        self._relative_change_history: list[float] = [
            10 * self._minimum_relative_change for _ in range(0, self._allowed_consecutive_violations + 1)
        ]
        self._last_population_evaluation: Optional[BasePopulationEvaluationResult] = None

    def reset_state(self) -> None:
        self._relative_change_history = [
            10 * self._minimum_relative_change for _ in range(0, self._allowed_consecutive_violations + 1)
        ]
        self._last_population_evaluation = None

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:

        if self._last_population_evaluation is not None:
            hausdorff_distance: float = _median_hausdorff_distance_by_expectation_value(
                self._last_population_evaluation, population_evaluation
            )
            best_individual_distance: float = abs(
                self._last_population_evaluation.best_expectation_value - population_evaluation.best_expectation_value
            )
            distance: float = max(hausdorff_distance, best_individual_distance)
            last_population_median_expectation: float = cast(
                float,
                median(
                    [
                        expectation
                        for expectation in self._last_population_evaluation.expectation_values
                        if expectation is not None
                    ]
                ),
            )

            self._relative_change_history.append(distance / last_population_median_expectation)

        self._last_population_evaluation = population_evaluation

        if len(self._relative_change_history) < self._allowed_consecutive_violations + 1:
            return False

        if (
            max(self._relative_change_history[-(self._allowed_consecutive_violations + 1) :])
            < self._minimum_relative_change
        ):
            return True

        return False
