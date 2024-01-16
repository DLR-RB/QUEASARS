# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Optional

import pytest
from dask.distributed import LocalCluster, Client
from docplex.mp.model import Model
from qiskit_aer.primitives import Estimator
from qiskit_algorithms.optimizers import NFT, Optimizer
from qiskit_optimization.translators import from_docplex_mp, to_ising
from qiskit_optimization.converters import IntegerToBinary
from qiskit.quantum_info import Operator

from queasars.circuit_evaluation.circuit_evaluation import OperatorCircuitEvaluator
from queasars.minimum_eigensolvers.base.evolutionary_algorithm import OperatorContext, BasePopulationEvaluationResult
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import EVQEIndividual
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.population import EVQEPopulation
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.mutation import (
    EVQELastLayerParameterSearch,
    EVQEParameterSearch,
    EVQETopologicalSearch,
    EVQELayerRemoval,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.speciation import EVQESpeciation
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.selection import EVQESelection


class TestEVQEOperators:
    client: Optional[Client] = None
    evaluator: Optional[OperatorCircuitEvaluator] = None
    operator: Optional[Operator] = None

    @pytest.fixture
    def initial_population(self) -> EVQEPopulation:
        return EVQEPopulation.random_population(
            n_qubits=4, n_layers=2, n_individuals=10, randomize_parameter_values=False, random_seed=0
        )

    @pytest.fixture
    def dask_client(self) -> Client:
        if self.client is None:
            cluster = LocalCluster(processes=True, n_workers=2)
            self.client = cluster.get_client()
        return self.client

    @pytest.fixture
    def hamiltonian(self) -> Operator:
        if self.operator is None:
            optimization_problem = Model()
            x = optimization_problem.integer_var(lb=0, ub=3, name="x")
            y = optimization_problem.integer_var(lb=0, ub=3, name="y")
            optimization_problem.minimize(x**2 - y**2)

            quadratic_program = from_docplex_mp(model=optimization_problem)
            integer_converter = IntegerToBinary()
            quadratic_program = integer_converter.convert(problem=quadratic_program)
            self.operator, _ = to_ising(quad_prog=quadratic_program)
        return self.operator

    @pytest.fixture
    def circuit_evaluator(self, hamiltonian) -> OperatorCircuitEvaluator:
        if self.evaluator is None:
            estimator = Estimator()
            estimator.set_options(seed=0)
            self.evaluator = OperatorCircuitEvaluator(qiskit_primitive=estimator, operator=hamiltonian)
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

    def test_topological_search_mutation(self, initial_population, operator_context):
        topological_search = EVQETopologicalSearch(mutation_probability=0.5, random_seed=0)

        initial_individual_length = sum(len(individual.layers) for individual in initial_population.individuals)
        new_population = topological_search.apply_operator(
            population=initial_population, operator_context=operator_context
        )
        new_individual_length = sum(len(individual.layers) for individual in new_population.individuals)

        assert initial_individual_length < new_individual_length

    def test_layer_removal_mutation(self, initial_population, operator_context):
        layer_removal = EVQELayerRemoval(mutation_probability=0.5, random_seed=0)

        initial_individual_length = sum(len(individual.layers) for individual in initial_population.individuals)
        new_population = layer_removal.apply_operator(population=initial_population, operator_context=operator_context)
        new_individual_length = sum(len(individual.layers) for individual in new_population.individuals)

        assert initial_individual_length > new_individual_length

    def test_speciation(self, initial_population, operator_context, optimizer):
        genetic_distance = 2
        last_layer_parameter_search = EVQELastLayerParameterSearch(
            mutation_probability=1, optimizer=optimizer, optimizer_n_circuit_evaluations=40
        )
        topological_search = EVQETopologicalSearch(mutation_probability=1, random_seed=0)
        speciation = EVQESpeciation(genetic_distance_threshold=genetic_distance, random_seed=0)
        selection = EVQESelection(alpha_penalty=0.1, beta_penalty=0.1, random_seed=0)

        population = last_layer_parameter_search.apply_operator(
            population=initial_population, operator_context=operator_context
        )
        population = speciation.apply_operator(population=population, operator_context=operator_context)
        population = selection.apply_operator(population=population, operator_context=operator_context)
        population = topological_search.apply_operator(population=population, operator_context=operator_context)
        population = last_layer_parameter_search.apply_operator(
            population=population, operator_context=operator_context
        )
        population = speciation.apply_operator(population=population, operator_context=operator_context)

        assert population.species_representatives is not None
        assert population.species_members is not None
        assert population.species_membership is not None

        for representative in population.species_representatives:
            for member_index in population.species_members[representative]:
                if representative != population.individuals[member_index]:
                    assert (
                        EVQEIndividual.get_genetic_distance(
                            individual_1=representative, individual_2=population.individuals[member_index]
                        )
                        < genetic_distance
                    )

    def test_selection(self, initial_population, operator_context, optimizer):
        context = operator_context
        evaluation_results: list[BasePopulationEvaluationResult] = []

        def callback(result: BasePopulationEvaluationResult):
            nonlocal evaluation_results
            evaluation_results.append(result)

        context.result_callback = callback

        layer_optimization = EVQELastLayerParameterSearch(
            mutation_probability=1, optimizer=optimizer, optimizer_n_circuit_evaluations=40, random_seed=0
        )
        speciation = EVQESpeciation(genetic_distance_threshold=2, random_seed=0)
        selection = EVQESelection(alpha_penalty=0.1, beta_penalty=0.1, random_seed=0)

        population = layer_optimization.apply_operator(population=initial_population, operator_context=operator_context)
        for _ in range(0, 3):
            population = speciation.apply_operator(population=population, operator_context=operator_context)
            population = selection.apply_operator(population=population, operator_context=operator_context)

        for i in range(1, len(evaluation_results)):
            assert sum(evaluation_results[i - 1].expectation_values) > sum(evaluation_results[i].expectation_values)
