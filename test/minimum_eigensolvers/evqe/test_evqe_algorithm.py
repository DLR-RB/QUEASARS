# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from queasars.minimum_eigensolvers.base.termination_criteria import BestIndividualRelativeImprovementTolerance
from queasars.minimum_eigensolvers.evqe.evqe import EVQEMinimumEigensolver, EVQEMinimumEigensolverConfiguration

from docplex.mp.model import Model
from qiskit_aer.primitives import Estimator, Sampler
from qiskit_algorithms.optimizers import NFT
from qiskit_optimization.translators import from_docplex_mp, to_ising
from qiskit_optimization.converters import IntegerToBinary


class TestEVQEAlgorithm:
    def test_simple_example(self):
        optimization_problem = Model()
        x = optimization_problem.integer_var(lb=0, ub=3, name="x")
        y = optimization_problem.integer_var(lb=0, ub=3, name="y")
        optimization_problem.minimize(x**2 - y**2)

        quadratic_program = from_docplex_mp(model=optimization_problem)
        integer_converter = IntegerToBinary()
        quadratic_program = integer_converter.convert(problem=quadratic_program)
        hamiltonian, _ = to_ising(quad_prog=quadratic_program)

        estimator = Estimator()
        estimator.set_options(seed=0)
        sampler = Sampler()
        sampler.set_options(seed=0)

        optimizer = NFT(maxiter=40)
        termination_criterion = BestIndividualRelativeImprovementTolerance(minimum_relative_improvement=0.01)

        solver_configuration = EVQEMinimumEigensolverConfiguration(
            sampler=sampler,
            estimator=estimator,
            optimizer=optimizer,
            optimizer_n_circuit_evaluations=40,
            dask_client=None,
            max_generations=None,
            max_circuit_evaluations=None,
            termination_criterion=termination_criterion,
            random_seed=0,
            population_size=10,
            randomize_initial_population_parameters=False,
            speciation_genetic_distance_threshold=2,
            selection_alpha_penalty=0.1,
            selection_beta_penalty=0.1,
            parameter_search_probability=0.25,
            topological_search_probability=0.5,
            layer_removal_probability=0.5,
        )

        solver = EVQEMinimumEigensolver(configuration=solver_configuration)

        result = solver.compute_minimum_eigenvalue(operator=hamiltonian)
        quasi_distribution = result.eigenstate.binary_probabilities()

        max_probability = max(quasi_distribution.values())
        best_measurements = [
            bitstring for bitstring, probability in quasi_distribution.items() if probability == max_probability
        ]
        bitstring = best_measurements[0]

        bitlist = [int(char) for char in bitstring][::-1]
        converted_variables = integer_converter.interpret(bitlist)

        assert converted_variables[0] == 0 and converted_variables[1] == 3
