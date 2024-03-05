# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dask.distributed import LocalCluster
import pytest
from qiskit_aer.primitives import Estimator
from qiskit_algorithms.optimizers import NFT, Optimizer
from qiskit.quantum_info import SparsePauliOp

from test.minimum_eigensolvers.evqe.model import create_sample_model, translate_model_to_hamiltonian
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


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=2)
    client = cluster.get_client()
    yield client
    cluster.close()


class TestEVQEOperators:

    @pytest.fixture
    def initial_population(self) -> EVQEPopulation:
        return EVQEPopulation.random_population(
            n_qubits=4, n_layers=2, n_individuals=10, randomize_parameter_values=False, random_seed=0
        )

    @pytest.fixture
    def hamiltonian(self) -> SparsePauliOp:
        model = create_sample_model()
        hamiltonian, _ = translate_model_to_hamiltonian(model=model)
        return hamiltonian

    @pytest.fixture
    def circuit_evaluator(self, hamiltonian) -> OperatorCircuitEvaluator:
        estimator = Estimator(approximation=True)
        estimator.set_options(seed=0)
        return OperatorCircuitEvaluator(qiskit_primitive=estimator, operator=hamiltonian)

    @pytest.fixture
    def optimizer(self) -> Optimizer:
        return NFT(maxfev=40)

    @pytest.fixture
    def operator_context(self, circuit_evaluator, dask_client) -> OperatorContext:
        return OperatorContext(
            circuit_evaluator=circuit_evaluator,
            dask_client=dask_client,
            result_callback=lambda x: None,
            circuit_evaluation_count_callback=lambda x: None,
        )

    def test_optimize_last_layer_mutation(self, initial_population, optimizer, circuit_evaluator, dask_client):
        evaluation_results: list[BasePopulationEvaluationResult] = []

        def callback(result: BasePopulationEvaluationResult):
            nonlocal evaluation_results
            evaluation_results.append(result)

        operator_context = OperatorContext(
            circuit_evaluator=circuit_evaluator,
            dask_client=dask_client,
            result_callback=callback,
            circuit_evaluation_count_callback=lambda x: None,
        )

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

        assert sum(evaluation_results[1].expectation_values) < sum(
            evaluation_results[0].expectation_values
        ), "Last layer optimization did not improve the expectation values on average!"

    def test_parameter_search_mutation(self, initial_population, optimizer, circuit_evaluator, dask_client):
        evaluation_results: list[BasePopulationEvaluationResult] = []

        def callback(result: BasePopulationEvaluationResult):
            nonlocal evaluation_results
            evaluation_results.append(result)

        operator_context = OperatorContext(
            circuit_evaluator=circuit_evaluator,
            dask_client=dask_client,
            result_callback=callback,
            circuit_evaluation_count_callback=lambda x: None,
        )

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

        assert sum(evaluation_results[1].expectation_values) < sum(
            evaluation_results[0].expectation_values
        ), "Parameter search did not improve the expectation values on average!"

    def test_topological_search_mutation(self, initial_population, operator_context):
        topological_search = EVQETopologicalSearch(mutation_probability=0.5, random_seed=0)

        initial_individual_length = sum(len(individual.layers) for individual in initial_population.individuals)
        new_population = topological_search.apply_operator(
            population=initial_population, operator_context=operator_context
        )
        new_individual_length = sum(len(individual.layers) for individual in new_population.individuals)

        assert (
            initial_individual_length < new_individual_length
        ), "Topological search did not increase the amount of layers in the population!"

    def test_layer_removal_mutation(self, initial_population, operator_context):
        layer_removal = EVQELayerRemoval(mutation_probability=0.5, random_seed=0)

        initial_individual_length = sum(len(individual.layers) for individual in initial_population.individuals)
        new_population = layer_removal.apply_operator(population=initial_population, operator_context=operator_context)
        new_individual_length = sum(len(individual.layers) for individual in new_population.individuals)

        assert initial_individual_length > new_individual_length

    def test_speciation(self, initial_population, optimizer, operator_context):
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
                    ), (
                        "A member of a species has exceeded the genetic_distance it should at "
                        + "maximum have to its species representative!"
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
            assert sum(evaluation_results[i - 1].expectation_values) > sum(
                evaluation_results[i].expectation_values
            ), "Selection did not improve the expectation values in the population on average!"
