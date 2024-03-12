# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass
from typing import Optional
from random import Random

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BasePopulation
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import EVQEIndividual
from queasars.utility.random import new_random_seed


@dataclass
class EVQEPopulation(BasePopulation[EVQEIndividual]):
    """Dataclass representing a population within the EVQE algorithm. The population consists of EVQEIndividuals
    each of which belongs to a species. Each species is represented by a member of the population. It is a member
    of its own species. If no information on speciation is available, species_representatives and species_members
    are both None

    :param species_representatives: list of species representatives. Its cardinality gives the amount of species
    :type species_representatives: Optional[set[EVQEIndividual]]
    :param species_members: dictionary mapping the species_representatives to all indices of the members of its species
    :type species_members: Optional[dict[EVQEIndividual, list[int]]
    :param species_membership: dictionary mapping each individual's index to its species representative
    :type species_membership: Optional[dict[int, int]
    """

    species_representatives: Optional[list[EVQEIndividual]]
    species_members: Optional[dict[EVQEIndividual, list[int]]]
    species_membership: Optional[dict[int, EVQEIndividual]]

    @staticmethod
    def random_population(
        n_qubits: int,
        n_layers: int,
        n_individuals: int,
        randomize_parameter_values: bool,
        random_seed: Optional[int] = None,
    ):
        """
        Generates a random population of n_individuals EVQEIndividuals with
        each one acting on n_qubits and having n_circuit_layers. The initial parameter values may either be
        initialized randomly or at 0. The species information in species_representatives and species_members is
        initialized as none.

        :arg n_qubits: amount of qubits on which the individuals shall act
        :type n_qubits: int
        :arg n_layers: amount of circuit layers each individual shall have
        :type n_layers: int
        :arg n_individuals: amount of individuals wanted for the generated population
        :type n_individuals: int
        :arg randomize_parameter_values: dictated whether parameter values shall be initialized randomly or at 0
        :type randomize_parameter_values: bool
        :arg random_seed: integer seed value to control randomness
        :type random_seed: Optional[int]
        :return: the generated EVQEPopulation
        :rtype: EVQEPopulation
        """
        random_generator = Random(random_seed)

        # Initialize random individuals
        individuals: tuple[EVQEIndividual, ...] = tuple(
            EVQEIndividual.random_individual(
                n_qubits=n_qubits,
                n_layers=n_layers,
                randomize_parameter_values=randomize_parameter_values,
                random_seed=new_random_seed(random_generator),
            )
            for _ in range(0, n_individuals)
        )

        return EVQEPopulation(
            individuals=individuals,
            species_representatives=None,
            species_members=None,
            species_membership=None,
        )
