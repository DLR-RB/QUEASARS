# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from typing import Optional

import pytest
from dask.distributed import LocalCluster, Client
from qiskit.quantum_info.random import random_hermitian
from qiskit_aer.primitives import Estimator
from qiskit_algorithms.optimizers import NFT, Optimizer

from queasars.circuit_evaluation.circuit_evaluation import OperatorCircuitEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import OperatorContext, BasePopulationEvaluationResult
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.population import EVQEPopulation
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.mutation import (
    EVQELastLayerParameterSearch,
    EVQEParameterSearch,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.speciation import EVQESpeciation
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.selection import EVQESelection


class TestEVQEMutations:
    client: Optional[Client] = None
    evaluator: Optional[OperatorCircuitEvaluator] = None

    @pytest.fixture
    def initial_population(self) -> EVQEPopulation:
        return EVQEPopulation.random_population(
            n_qubits=2, n_layers=2, n_individuals=25, randomize_parameter_values=False, random_seed=0
        )

    @pytest.fixture
    def dask_client(self) -> Client:
        if self.client is None:
            cluster = LocalCluster(processes=True)
            self.client = cluster.get_client()
        return self.client

    @pytest.fixture
    def circuit_evaluator(self) -> OperatorCircuitEvaluator:
        if self.evaluator is None:
            hermitian = random_hermitian(dims=4, seed=0)
            estimator = Estimator()
            estimator.set_options(seed=0)
            self.evaluator = OperatorCircuitEvaluator(qiskit_primitive=estimator, operator=hermitian)
        return self.evaluator

    @pytest.fixture
    def operator_context(self, dask_client, circuit_evaluator) -> OperatorContext:
        return OperatorContext(
            circuit_evaluator=circuit_evaluator,
            dask_client=dask_client,
            result_callback=lambda x: None,
            circuit_evaluation_count_callback=lambda x: None,
        )

    @pytest.fixture
    def optimizer(self) -> Optimizer:
        return NFT(maxfev=40)

    def test_optimize_last_layer_mutation(self, initial_population, operator_context, optimizer):
        context = operator_context
        evaluation_results: list[BasePopulationEvaluationResult] = []

        def callback(result: BasePopulationEvaluationResult):
            nonlocal evaluation_results
            evaluation_results.append(result)

        context.result_callback = callback

        last_layer_search = EVQELastLayerParameterSearch(
            mutation_probability=0.3, optimizer=optimizer, optimizer_n_circuit_evaluations=40, random_seed=0
        )
        speciation = EVQESpeciation(genetic_distance_threshold=3, random_seed=0)
        evaluation = EVQESelection(alpha_penalty=0.1, beta_penalty=0.1, random_seed=0)

        population = initial_population
        population = speciation.apply_operator(population=population, operator_context=operator_context)
        _ = evaluation.apply_operator(population=population, operator_context=operator_context)
        population = last_layer_search.apply_operator(population=population, operator_context=operator_context)
        population = speciation.apply_operator(population=population, operator_context=operator_context)
        _ = evaluation.apply_operator(population=population, operator_context=operator_context)

        assert sum(evaluation_results[1].expectation_values) < sum(evaluation_results[0].expectation_values)

    def test_parameter_search_mutation(self, initial_population, operator_context, optimizer):
        context = operator_context
        evaluation_results: list[BasePopulationEvaluationResult] = []

        def callback(result: BasePopulationEvaluationResult):
            nonlocal evaluation_results
            evaluation_results.append(result)

        context.result_callback = callback

        parameter_search = EVQEParameterSearch(
            mutation_probability=0.3, optimizer=optimizer, optimizer_n_circuit_evaluations=40, random_seed=0
        )
        speciation = EVQESpeciation(genetic_distance_threshold=3, random_seed=0)
        evaluation = EVQESelection(alpha_penalty=0.1, beta_penalty=0.1, random_seed=0)

        population = initial_population
        population = speciation.apply_operator(population=population, operator_context=operator_context)
        _ = evaluation.apply_operator(population=population, operator_context=operator_context)
        population = parameter_search.apply_operator(population=population, operator_context=operator_context)
        population = speciation.apply_operator(population=population, operator_context=operator_context)
        _ = evaluation.apply_operator(population=population, operator_context=operator_context)

        assert sum(evaluation_results[1].expectation_values) < sum(evaluation_results[0].expectation_values)