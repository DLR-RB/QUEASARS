# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Optional
from numpy.typing import NDArray


class SPSATerminationChecker:
    """
    Termination checker for qiskit_algorithms SPSA optimizer. For each iteration, it checks that the
    absolute difference between the last and new function value, divided by the last function value,
    does not fall below minimum_relative_change for more than allowed_consecutive_violations times.

    :param minimum_relative_change: threshold value for the change in function value relative to the last
        function value, below which this termination checker chooses to terminate the SPSA optimization
    :type minimum_relative_change: float
    :param allowed_consecutive_violations: determines how often the threshold value can be violated consecutively
        before this termination checker chooses to terminate. If set to 0, this terminates the first time the change
        falls below the threshold value. If set to 2 for example, this terminates the first time the change falls below
        the threshold three consecutive times. Must be at least 0.
    :type allowed_consecutive_violations: int
    """

    def __init__(self, minimum_relative_change: float, allowed_consecutive_violations: int):
        self._minimum_relative_change: float = minimum_relative_change
        self._allowed_consecutive_violations: int = allowed_consecutive_violations
        self._function_value_history: list[float] = []
        self._change_history: list[float] = []
        self._n_function_evaluations = 0
        self._best_function_value: float = float("inf")
        self._best_parameter_values: Optional[NDArray] = None
        self._done = False

    def termination_check(
        self,
        n_function_evaluations: int,
        parameter_values: NDArray,
        function_value: float,
        step_size: float,
        accepted: bool,
    ) -> bool:
        """Given the callback values provided by qiskit_algorithm's SPSA optimizer, this method determines
        whether the SPSA optimization should terminate"""

        if self._done:
            self._function_value_history = []
            self._change_history = []
            self._n_function_evaluations = 0
            self._best_function_value = float("inf")
            self._best_parameter_values = None
            self._done = False

        self._n_function_evaluations = n_function_evaluations
        self._function_value_history.append(function_value)

        if function_value < self._best_function_value:
            self._best_function_value = function_value
            self._best_parameter_values = parameter_values

        if len(self._function_value_history) < 2:
            return False

        change = abs(function_value - self._function_value_history[-2]) / self._function_value_history[-2]
        self._change_history.append(change)

        if len(self._change_history) < self._allowed_consecutive_violations + 1:
            return False

        if max(self._change_history[-self._allowed_consecutive_violations - 1 :]) < self._minimum_relative_change:
            self._done = True
            return True

        return False

    @property
    def n_function_evaluations(self) -> int:
        """
        :return: the amount of function evaluations used by SPSA until termination
        :rtype: int
        """
        return self._n_function_evaluations

    @property
    def function_value_history(self) -> list[float]:
        """
        :return: a list of all encountered function values
        :rtype: list[float]
        """
        return self._function_value_history

    @property
    def best_function_value(self) -> float:
        """
        :return: the best function value encountered during the optimization
        :rtype: float
        """
        return self._best_function_value

    @property
    def best_parameter_values(self) -> NDArray:
        """
        :return: the best parameter values found during the optimization
        :rtype: NDArray
        :raises: ValueError, if termination_check was never called
        """
        if self._best_parameter_values is None:
            raise ValueError(
                "The termination checker seems to have never been called! Therefore it currently"
                + "stores no parameter values!"
            )

        return self._best_parameter_values