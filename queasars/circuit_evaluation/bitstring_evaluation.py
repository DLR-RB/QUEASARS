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
        self._input_length: int = input_length
        self._evaluation_function: Callable[[str], float] = evaluation_function

    def _check_bitstring(self, bitstring: str) -> None:
        """
        Raises an exception if the bitstring is invalid, therefore if it is not of the length
        self._input_length or if it contains characters other than '0' or '1'.

        :arg bitstring: to check for validity in the context of this BitstringEvaluator
        :type bitstring: str
        :raises: BitstringEvaluatorException
        """
        if len(bitstring) != self._input_length:
            raise BitstringEvaluatorException(
                f"Bitstring must be of the length {self._input_length} " + f"but was of length {len(bitstring)}!"
            )

        if any(character not in ("0", "1") for character in bitstring):
            raise BitstringEvaluatorException("Bitstring may not contain characters other than '0' or '1'!")

    def evaluate_bitstring(self, bitstring: str) -> float:
        """Applies the evaluation_function to the given bitstring and returns the resulting float.
        :arg bitstring: Bitstring to apply the the evaluation_function to
        :type bitstring: str
        :raises: BitstringEvaluatorException: If the given bitstring does not match input_length,
            if it contains characters other than 0 or 1, or if the evaluation_function itself fails.
        :return: Floating point number returned by the evaluation_function.
        :rtype: float
        """

        self._check_bitstring(bitstring=bitstring)
        return self._evaluation_function(bitstring)

    @property
    def input_length(self):
        """Gets the length of the bitstrings on which this BitstringEvaluator may act

        :return: length of the bitstrings on which this BistringEvaluator may act
        :rtype: int
        """
        return self._input_length


class BitstringEvaluatorException(Exception):
    """Class for exceptions caused during the bitstring evaluation."""
