# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from random import Random


def new_random_seed(random_generator: Random) -> int:
    """Generate a new random integer seed in the range (0, 2147483647)

    :arg random_generator: used to generate a new integer seed
    :type random_generator: Random
    :return: a new random seed
    :rtype: int
    """
    return random_generator.randint(0, 2147483647)
