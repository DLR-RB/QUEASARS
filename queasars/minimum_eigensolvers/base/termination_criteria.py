# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod

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


class BestIndividualAbsoluteImprovementTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):
    """Termination criterion which terminates when the absolute change in expectation value between the best individual
    of the current and the last generation falls below an absolute threshold value

    :param minimum_absolute_improvement: expectation value improvement below which the algorithm shall terminate.
        Must be bigger than 0
    :type minimum_absolute_improvement: float
    """

    def __init__(self, minimum_absolute_improvement: float):
        if minimum_absolute_improvement <= 0:
            raise ValueError("The minimum absolute improvement parameter must be bigger than 0!")

        self.minimum_absolute_improvement: float = minimum_absolute_improvement
        self.previous_best_expectation_value: float = float("inf")

    def reset_state(self) -> None:
        self.previous_best_expectation_value = float("inf")

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:
        if (
            population_evaluation.best_expectation_value
            < self.previous_best_expectation_value - self.minimum_absolute_improvement
        ):
            self.previous_best_expectation_value = population_evaluation.best_expectation_value
            return False

        return True


class BestIndividualRelativeImprovementTolerance(EvolvingAnsatzMinimumEigensolverBaseTerminationCriterion):
    """Termination criterion which terminates when the change in expectation value between
    the best individual of the current and last generation falls below a threshold. This threshold is taken
    as a percentage of the expectation value of the best individual of the last generation

    :param minimum_relative_improvement: relative improvement in expectation value below which the algorithm shall
        terminate. Must be in the range )0,1)
    :type minimum_relative_improvement: float
    """

    def __init__(self, minimum_relative_improvement: float):
        if minimum_relative_improvement <= 0 or minimum_relative_improvement > 1:
            raise ValueError("The minimum relative improvement parameter must not exceed the range )0,1)!")

        self.minimum_relative_improvement = minimum_relative_improvement
        self.previous_best_expectation_value: float = float("inf")

    def reset_state(self) -> None:
        self.previous_best_expectation_value = float("inf")

    def check_termination(
        self,
        population_evaluation: BasePopulationEvaluationResult,
        best_individual: BaseIndividual,
        best_expectation_value: float,
    ) -> bool:
        if self.previous_best_expectation_value == float("inf"):
            self.previous_best_expectation_value = population_evaluation.best_expectation_value
            return False
        if population_evaluation.best_expectation_value < self.previous_best_expectation_value - abs(
            self.minimum_relative_improvement * self.previous_best_expectation_value
        ):
            self.previous_best_expectation_value = population_evaluation.best_expectation_value
            return False

        return True
