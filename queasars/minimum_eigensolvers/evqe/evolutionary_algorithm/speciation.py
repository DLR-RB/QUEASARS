# -*- coding: utf-8 -*-
# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from random import choice

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
    """

    def __init__(self, genetic_distance_threshold: int):
        """Constructor method"""
        self.genetic_distance_threshold = genetic_distance_threshold

    def apply_operator(self, population: EVQEPopulation, operator_context: OperatorContext) -> EVQEPopulation:
        species_representatives: set[EVQEIndividual]
        if population.species_representatives is None:
            species_representatives = set()
        else:
            species_representatives = population.species_representatives
        species_members: dict[EVQEIndividual, list[int]] = {}

        # assign each individual to a species
        for i, individual in enumerate(population.individuals):
            found_species: bool = False

            # search a representative with a sufficiently low genetic distance to the individual
            for representative in species_representatives:
                if (
                    EVQEIndividual.get_genetic_distance(individual_1=individual, individual_2=representative)
                    < self.genetic_distance_threshold
                ):
                    # if such a representative was found add the individual to the representative's species
                    species_members[representative].append(i)
                    found_species = True

            # if no close representative was found, make the individual its own representative, creating a new species
            if not found_species:
                species_representatives.add(individual)
                species_members[individual] = [i]

        # after species assignment draw new random species representatives
        species_members = {population.individuals[choice(members)]: members for members in species_members.values()}
        species_representatives = set(species_members.keys())

        return EVQEPopulation(
            individuals=population.individuals,
            species_representatives=species_representatives,
            species_members=species_members,
        )
