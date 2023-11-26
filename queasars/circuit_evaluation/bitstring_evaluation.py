# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Callable


class BitstringEvaluator:
    def __init__(self, input_length: int, evaluation_function: Callable[[str], float]):
        pass

    def evaluate_bitstring(self, bitstring: str) -> float:
        pass
