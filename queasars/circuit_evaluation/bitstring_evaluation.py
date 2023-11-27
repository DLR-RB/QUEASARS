# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Callable


class BitstringEvaluator:
    """Wrapper class for functions which map bitstrings of fixed length to floating point numbers.

    :param input_length: Length of the bitstrings the evaluation_function supports
    :type input_length: int
    :param evaluation_function: Function mapping bitstrings to floating point numbers
    :type evaluation_function: Callable[[str], float]"""

    def __init__(self, input_length: int, evaluation_function: Callable[[str], float]):
        """Constructor method"""

    def evaluate_bitstring(self, bitstring: str) -> float:
        """Applies the evaluation_function to the given bitstring and returns the resulting float.
        :param bitstring: Bitstring to apply the the evaluation_function to
        :type bitstring: str
        :raises BitstringEvaluatorException: If the given bitstring does not match input_length,
            if it contains characters other than 0 or 1, or if the evaluation_function itself fails.
        :return: Floating point number returned by the evaluation_function.
        :rtype: float
        """
        raise NotImplementedError


class BitstringEvaluatorException(Exception):
    """Class for exceptions caused during the bitstring evaluation."""
