# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dask.distributed import Client
from typing import Optional
from qiskit_aer.primitives import Estimator, Sampler
from qiskit_algorithms.optimizers import NFT

from queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver import (
    EvolvingAnsatzMinimumEigensolverResult,
)
from queasars.minimum_eigensolvers.base.termination_criteria import BestIndividualRelativeChangeTolerance
from queasars.minimum_eigensolvers.evqe.evqe import EVQEMinimumEigensolver, EVQEMinimumEigensolverConfiguration


def create_sample_solver(client: Optional[Client] = None, mutually_exclusive_primitives: bool = True) -> EVQEMinimumEigensolver:
    estimator = Estimator(approximation=True)
    sampler = Sampler()
    optimizer = NFT(maxiter=40)
    termination_criterion = BestIndividualRelativeChangeTolerance(minimum_relative_change=0.005)

    solver_configuration = EVQEMinimumEigensolverConfiguration(
        sampler=sampler,
        estimator=estimator,
        optimizer=optimizer,
        optimizer_n_circuit_evaluations=40,
        max_generations=None,
        max_circuit_evaluations=None,
        termination_criterion=termination_criterion,
        random_seed=0,
        population_size=10,
        randomize_initial_population_parameters=False,
        speciation_genetic_distance_threshold=3,
        selection_alpha_penalty=0.1,
        selection_beta_penalty=0.1,
        parameter_search_probability=0.24,
        topological_search_probability=0.2,
        layer_removal_probability=0.05,
        parallel_executor=client,
        mutually_exclusive_primitives=mutually_exclusive_primitives,
    )

    return EVQEMinimumEigensolver(configuration=solver_configuration)


def get_likeliest_bitstrings_from_result(result: EvolvingAnsatzMinimumEigensolverResult) -> list[str]:
    quasi_distribution = result.eigenstate.binary_probabilities()
    max_probability = max(quasi_distribution.values())
    best_measurements = [
        bitstring for bitstring, probability in quasi_distribution.items() if probability == max_probability
    ]
    return best_measurements
