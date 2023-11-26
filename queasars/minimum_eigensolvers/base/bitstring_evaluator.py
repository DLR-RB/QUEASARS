# -*- coding: utf-8 -*-
# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Callable


class BitstringEvaluator:
    def __init__(self, input_length: int, evaluation_function: Callable[[str], float]):
        self.input_length: int = input_length
        self.evaluation_function: Callable[[str], float] = evaluation_function

    def evaluate_bitstring(self, bitstring: str) -> float:
        if len(bitstring) != self.input_length:
            raise ValueError(
                f"Bitstring to be evaluated needs to be of length {self.input_length}, but was {len(bitstring)}"
            )
        return self.evaluation_function(bitstring)
