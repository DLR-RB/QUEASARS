# -*- coding: utf-8 -*-
# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from random import Random
from typing import Optional

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import (
    BaseEvolutionaryOperator,
    OperatorContext,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import (
    EVQEIndividual,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.population import (
    EVQEPopulation,
)


class EVQESpeciation(BaseEvolutionaryOperator[EVQEPopulation]):
    """Class representing the speciation evolutionary operator within the EVQE algorithm.

    :param genetic_distance_threshold: maximum genetic distance between two individuals to classify
        them as belonging to different species
    :type genetic_distance_threshold: int
    :param random_seed: integer value to control randomness
    :type: Optional[int]
    """

    def __init__(self, genetic_distance_threshold: int, random_seed: Optional[int]):
        """Constructor method"""
        self.genetic_distance_threshold: int = genetic_distance_threshold
        self.random_generator: Random = Random(random_seed)

    def apply_operator(self, population: EVQEPopulation, operator_context: OperatorContext) -> EVQEPopulation:
        species_representatives: list[EVQEIndividual]
        if population.species_representatives is None:
            species_representatives = []
            species_members: dict[EVQEIndividual, list[int]] = {}
        else:
            species_representatives = population.species_representatives
            species_members = {representative: [] for representative in species_representatives}
        species_membership: dict[int, EVQEIndividual] = {}

        # assign each individual to a species
        for i, individual in enumerate(population.individuals):
            found_species: bool = False

            # search a representative with a sufficiently low genetic distance to the individual
            for representative in species_representatives:
                if (
                    EVQEIndividual.get_genetic_distance(individual_1=individual, individual_2=representative)
                    < self.genetic_distance_threshold
                    or individual == representative
                ):
                    # if such a representative was found add the individual to the representative's species
                    species_members[representative].append(i)
                    species_membership[i] = representative
                    found_species = True
                    break

            # if no close representative was found, make the individual its own representative, creating a new species
            if not found_species:
                species_representatives.append(individual)
                species_members[individual] = [i]
                species_membership[i] = individual

        # after species assignment draw new random species representatives
        new_species_members: dict[EVQEIndividual, list[int]] = {}
        for members in species_members.values():
            if len(members) <= 0:
                continue
            representative_index = self.random_generator.choice(members)
            representative = population.individuals[representative_index]
            if representative not in new_species_members:
                new_species_members[representative] = members
            else:
                new_species_members[representative].extend(members)

        species_membership = {}
        for representative, members in new_species_members.items():
            for member in members:
                species_membership[member] = representative
        species_representatives = list(new_species_members.keys())

        return EVQEPopulation(
            individuals=population.individuals,
            species_representatives=species_representatives,
            species_members=new_species_members,
            species_membership=species_membership,
        )

    def get_n_expected_circuit_evaluations(
        self, population: EVQEPopulation, operator_context: OperatorContext
    ) -> Optional[int]:
        return 0
