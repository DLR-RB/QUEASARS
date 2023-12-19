# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from typing import Optional
from random import Random

from dask.distributed import Future, wait

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import (
    BaseEvolutionaryOperator,
    OperatorContext,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import EVQEIndividual
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.population import EVQEPopulation


class EVQESelection(BaseEvolutionaryOperator[EVQEPopulation, OperatorContext]):
    """Class representing the selection evolutionary operator of the EVQE algorithm

    :param alpha_penalty: scaling factor for penalizing the amount of circuit layers of an individual
    :type: float
    :param beta_penalty: scaling factor for penalizing the amount of controlled gates of an individual
    :type: float
    :param random_seed: integer value to control randomness
    :type: int
    """

    def __init__(self, alpha_penalty: float, beta_penalty: float, random_seed: Optional[int]):
        self.alpha_penalty: float = alpha_penalty
        self.beta_penalty: float = beta_penalty
        self.random_generator: Random = Random(random_seed)

    def apply_operator(self, population: EVQEPopulation, operator_context: OperatorContext) -> EVQEPopulation:
        print("species size: ", len(population.individuals))

        future_circuit_evaluations: list[Future] = [
            operator_context.dask_client.submit(
                operator_context.circuit_evaluator.evaluate_circuits,
                [individual.get_parameterized_quantum_circuit()],
                [individual.get_parameter_values()],
            )
            for individual in population.individuals
        ]

        wait(future_circuit_evaluations)
        circuit_evaluations: list[float] = [future.result()[0] for future in future_circuit_evaluations]
        del future_circuit_evaluations

        offset: float = abs(min(circuit_evaluations)) + 1
        circuit_evaluations = [evaluation + offset for evaluation in circuit_evaluations]

        if (
            population.species_representatives is None
            or population.species_members is None
            or population.species_membership is None
        ):
            raise EVQESelectionException(
                "Selection can't be finished if speciation information is missing!\n"
                + "Either the species_representatives, species_members or species_membership\n"
                + "attribute of the population is None!"
            )

        fitnesses: list[float] = [
            (
                circuit_evaluations[i]
                + self.alpha_penalty * len(individual.layers)
                + self.beta_penalty * individual.get_n_controlled_gates()
            )
            / float(len(population.species_members[population.species_membership[i]]))
            for i, individual in enumerate(population.individuals)
        ]

        summed_fitness: float = sum(fitnesses)
        fitness_weights: list[float] = [1 / (fitness / summed_fitness) for fitness in fitnesses]

        selected_individuals: list[EVQEIndividual] = self.random_generator.choices(
            population.individuals, weights=fitness_weights, k=len(population.individuals)
        )

        return EVQEPopulation(
            individuals=tuple(selected_individuals),
            species_representatives=population.species_representatives,
            species_members=None,
            species_membership=None,
        )

    def get_n_expected_circuit_evaluations(
        self, population: EVQEPopulation, operator_context: OperatorContext
    ) -> Optional[int]:
        pass


class EVQESelectionException(Exception):
    """Class to represent errors caused during the EVQE selection operator"""
