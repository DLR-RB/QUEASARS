# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Optional

from qiskit.circuit import QuantumCircuit
from qiskit.result import QuasiDistribution
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BasePopulationEvaluationResult, BaseIndividual


class EvolvingAnsatzMinimumEigensolverResult(MinimumEigensolverResult):
    """Evolving ansatz minimum eigensolver result"""

    def __init__(self) -> None:
        super().__init__()
        self._eigenstate: Optional[QuasiDistribution] = None
        self._best_individual: Optional[BaseIndividual] = None
        self._circuit_evaluations: Optional[list[int]] = None
        self._generations: Optional[int] = None
        self._population_evaluation_results: Optional[list[BasePopulationEvaluationResult]] = None
        self._initial_state_circuit: Optional[QuantumCircuit] = None

    @property
    def eigenstate(self) -> Optional[QuasiDistribution]:
        """Return the quasi-distribution sampled from the final state

        :return: Quasi-distribution sampled from the final state
        :rtype: Optional[QuasiDistribution]
        """
        return self._eigenstate

    @eigenstate.setter
    def eigenstate(self, value: QuasiDistribution):
        """Set the the quasi-distribution sampled from the final state

        :arg value: Value to set the sampled quasi-distribution to
        :type value: QuasiDistribution
        """
        self._eigenstate = value

    @property
    def best_individual(self) -> Optional[BaseIndividual]:
        """Returns the best Individual encountered during the evolution

        :return: The best Individual encountered during the evolution
        :rtype: BaseIndividual
        """
        return self._best_individual

    @best_individual.setter
    def best_individual(self, value: BaseIndividual):
        """Sets the best Individual encountered during the evolution

        :arg value: Value to set the best Individual to
        :type value: BaseIndividual
        """
        self._best_individual = value

    @property
    def optimal_parameters(self) -> Optional[tuple[float, ...]]:
        """Returns the optimal parameters in a dictionary

        :return: The optimal parameters
        :rtype: Optional[tuple[float, ...]]
        """
        if self._best_individual is not None:
            return self._best_individual.get_parameter_values()
        return None

    @property
    def optimal_circuit(self) -> Optional[QuantumCircuit]:
        """The optimal circuit. Along with the optimal parameters,
        this can be used to retrieve the minimum eigenstate.

        :return: The optimal parameterized quantum circuit
        :rtype: Optional[QuantumCircuit]
        """
        if self._best_individual is not None:
            return self._best_individual.get_parameterized_quantum_circuit()
        return None

    @property
    def circuit_evaluations(self) -> Optional[list[int]]:
        """Returns the number of circuit evaluations used by the eigensolver per generation

        :return: The number of circuit evaluations per generation
        :rtype: Optional[list[int]]
        """
        return self._circuit_evaluations

    @circuit_evaluations.setter
    def circuit_evaluations(self, value: list[int]) -> None:
        """Sets the number of circuit evaluations used by the eigensolver per generation

        :arg value: Value to set the number of circuit evaluations per generation to
        :type value: list[int]
        """
        self._circuit_evaluations = value

    @property
    def generations(self) -> Optional[int]:
        """Returns the number of generations the evolutionary algorithm was run for

        :return: The number of generations the algorithm was run for
        :rtype: Optional[int]
        """
        return self._generations

    @generations.setter
    def generations(self, value: int):
        """Sets the number of generations the evolutionary algorithm was run for

        :arg value: Value to set the number of generations to
        :type value: int
        """
        self._generations = value

    @property
    def population_evaluation_results(self) -> Optional[list[BasePopulationEvaluationResult]]:
        """Returns the list of  all population evaluation results

        :return: The list of all population evaluation results gathered during the optimization
        :rtype: Optional[list[BasePopulationEvaluationResult]]
        """
        return self._population_evaluation_results

    @population_evaluation_results.setter
    def population_evaluation_results(self, value: list[BasePopulationEvaluationResult]):
        """Sets the evaluation results gathered during the optimization

        :arg value: Values to set the evaluation results to. Should be in order of their appearance
        :type value: list[BasePopulationEvaluationResult]
        """
        self._population_evaluation_results = value

    @property
    def final_population_evaluation_result(self) -> Optional[BasePopulationEvaluationResult]:
        """Returns the final population evaluation result

        :return: The final population evaluation result
        :rtype: Optional[BasePopulationEvaluationResult]
        """
        if self.population_evaluation_results is not None and len(self.population_evaluation_results) != 0:
            return self.population_evaluation_results[-1]
        return None

    @property
    def initial_state_circuit(self) -> Optional[QuantumCircuit]:
        """Returns the quantum circuit which was used to initialize all individuals in one initial state

        :return: the quantum circuit used to initialize all individuals in one initial state
        :rtype: Optional[QuantumCircuit]
        """
        return self._initial_state_circuit

    @initial_state_circuit.setter
    def initial_state_circuit(self, value: QuantumCircuit):
        """Sets the quantum circuit which was used to initialize all individuals in one initial state

        :arg value: the quantum circuit which was used to initialize all individuals in one initial state
        :type value: QuantumCircuit
        """
        self._initial_state_circuit = value
