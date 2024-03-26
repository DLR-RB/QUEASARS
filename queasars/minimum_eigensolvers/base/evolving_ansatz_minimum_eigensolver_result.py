# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Optional

from qiskit.circuit import QuantumCircuit
from qiskit.result import QuasiDistribution
from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BasePopulationEvaluationResult


class EvolvingAnsatzMinimumEigensolverResult(MinimumEigensolverResult):
    """Evolving ansatz minimum eigensolver result"""

    def __init__(self) -> None:
        super().__init__()
        self._eigenstate: Optional[QuasiDistribution] = None
        self._optimal_parameters: Optional[dict] = None
        self._optimal_circuit: Optional[QuantumCircuit] = None
        self._circuit_evaluations: Optional[int] = None
        self._generations: Optional[int] = None
        self._population_evaluation_results: Optional[list[BasePopulationEvaluationResult]] = None

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
    def optimal_parameters(self) -> Optional[dict]:
        """Returns the optimal parameters in a dictionary

        :return: The optimal parameters
        :rtype: Optional[dict]
        """
        return self._optimal_parameters

    @optimal_parameters.setter
    def optimal_parameters(self, value: dict) -> None:
        """Sets optimal parameters

        :arg value: Value to set the optimal parameters to
        :type value: dict
        """
        self._optimal_parameters = value

    @property
    def optimal_circuit(self) -> Optional[QuantumCircuit]:
        """The optimal circuit. Along with the optimal parameters,
        this can be used to retrieve the minimum eigenstate.

        :return: The optimal parameterized quantum circuit
        :rtype: Optional[QuantumCircuit]
        """
        return self._optimal_circuit

    @optimal_circuit.setter
    def optimal_circuit(self, value: QuantumCircuit) -> None:
        """Sets the optimal circuit

        :arg value: Value to set the optimal circuit to
        :type value: QuantumCircuit
        """
        self._optimal_circuit = value

    @property
    def circuit_evaluations(self) -> Optional[int]:
        """Returns the number of circuit evaluations used by the eigensolver

        :return: The number of circuit evaluations
        :rtype: int
        """
        return self._circuit_evaluations

    @circuit_evaluations.setter
    def circuit_evaluations(self, value: int) -> None:
        """Sets the number of circuit evaluations used by the eigensolver

        :arg value: Value to set the number of circuit evaluations to
        :type: int
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
