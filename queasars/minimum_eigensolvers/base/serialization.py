# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from base64 import b64encode, b64decode
from io import BytesIO
from json import JSONEncoder, JSONDecoder
from typing import Any

from qiskit.circuit import QuantumCircuit
from qiskit.qpy import dump as qpy_dump, load as qpy_load
from qiskit.result import QuasiDistribution

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BasePopulationEvaluationResult
from queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver_result import (
    EvolvingAnsatzMinimumEigensolverResult,
)
from queasars.minimum_eigensolvers.evqe.serialization import EVQEPopulationJSONEncoder, EVQEPopulationJSONDecoder


class EvolvingAnsatzMinimumEigensolverResultJSONEncoder(JSONEncoder):
    """
    JSONEncoder class for encoding EvolvingAnsatzMinimumEigensolverResult instances as JSON.
    This class can serialize the following QUEASARS classes:
        BasePopulationEvaluationResult
        EvolvingAnsatzMinimumEigensolverResult
    It can also serialize the following Qiskit classes:
        QuasiDistribution
        QuantumCircuit
    Finally, it can serialize python's complex numbers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evqe_encoder = EVQEPopulationJSONEncoder(*args, **kwargs)

    def default(self, o: Any):

        if any(isinstance(o, t) for t in self._evqe_encoder.serializable_types()):
            return self._evqe_encoder.default(o)

        if o is None:
            return None

        if isinstance(o, complex):
            return {
                "complex_number_real_value": float(o.real),
                "complex_number_imaginary_value": float(o.imag),
            }

        if isinstance(o, QuasiDistribution):
            return {
                "quasidistribution_data": [[key, value] for key, value in o.items()],
                "quasidistribution_shots": o.shots,
                "quasidistribution_stdev_bound": o.stddev_upper_bound,
            }

        if isinstance(o, QuantumCircuit):
            buffer = BytesIO()
            qpy_dump(programs=o, file_obj=buffer)
            str_encoded = b64encode(buffer.getvalue()).decode("utf-8")
            return {"qiskit_quantum_circuit": str_encoded}

        if isinstance(o, BasePopulationEvaluationResult):
            return {
                "base_population_evaluation_population": self.default(o.population),
                "base_population_evaluation_expectation_values": list(o.expectation_values),
                "base_population_evaluation_best_individual": self.default(o.best_individual),
                "base_population_evaluation_best_expectation_value": o.best_expectation_value,
            }

        if isinstance(o, EvolvingAnsatzMinimumEigensolverResult):

            if isinstance(o.eigenvalue, complex):
                eigenvalue = self.default(o.eigenvalue)
            else:
                eigenvalue = o.eigenvalue

            if isinstance(o.aux_operators_evaluated, list):
                aux_operators_evaluated = {
                    "type": "list",
                    "values": [self.default(value) for value in o.aux_operators_evaluated],
                }
            elif isinstance(o.aux_operators_evaluated, dict):
                aux_operators_evaluated = {
                    "type": "dict",
                    "values": [[key, self.default(value)] for key, value in o.aux_operators_evaluated.items()],
                }
            else:
                aux_operators_evaluated = None

            if isinstance(o.population_evaluation_results, list):
                population_evaluation_results = [self.default(res) for res in o.population_evaluation_results]
            else:
                population_evaluation_results = None

            return {
                "evolving_ansatz_result_eigenvalue": eigenvalue,
                "evolving_ansatz_result_aux_operators_evaluated": aux_operators_evaluated,
                "evolving_ansatz_result_eigenstate": self.default(o.eigenstate),
                "evolving_ansatz_result_best_individual": self.default(o.best_individual),
                "evolving_ansatz_result_circuit_evaluations": o.circuit_evaluations,
                "evolving_ansatz_result_generations": o.generations,
                "evolving_ansatz_population_evaluation_results": population_evaluation_results,
                "evolving_ansatz_population_initial_state_circuit": self.default(o.initial_state_circuit),
            }

    @staticmethod
    def serializable_types() -> set[type]:
        """
        :return: a set of all types, which this encoder can serialize
        :rtype: set[type]
        """
        return {
            complex,
            QuasiDistribution,
            QuantumCircuit,
            BasePopulationEvaluationResult,
            EvolvingAnsatzMinimumEigensolverResult,
        }


class EvolvingAnsatzMinimumEigensolverResultJSONDecoder(JSONDecoder):
    """
    JSONDecoder class for decoding EvolvingAnsatzMinimumEigensolverResult instances from JSON.
    This class can deserialize the following QUEASARS classes:
        BasePopulationEvaluationResult
        EvolvingAnsatzMinimumEigensolverResult
    It can also deserialize the following Qiskit classes:
        QuasiDistribution
        QuantumCircuit
    Finally, it can deserialize Python's complex numbers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
        self._evqe_population_decoder = EVQEPopulationJSONDecoder(*args, **kwargs)

    @staticmethod
    def identifying_keys() -> set[str]:
        """
        :return: a set of all keys, which the object_hook of this decoder can identify
        :rtype: set[str]
        """
        return {
            "complex_number_real_value",
            "complex_number_imaginary_value",
            "quasidistribution_data",
            "quasidistribution_shots",
            "quasidistribution_stdev_bound",
            "qiskit_quantum_circuit",
            "base_population_evaluation_population",
            "base_population_evaluation_expectation_values",
            "base_population_evaluation_best_individual",
            "base_population_evaluation_best_expectation_value",
            "evolving_ansatz_result_eigenvalue",
            "evolving_ansatz_result_aux_operators_evaluated",
            "evolving_ansatz_result_eigenstate",
            "evolving_ansatz_result_best_individual",
            "evolving_ansatz_result_circuit_evaluations",
            "evolving_ansatz_result_generations",
            "evolving_ansatz_population_evaluation_results",
            "evolving_ansatz_population_initial_state_circuit",
        }.union(EVQEPopulationJSONDecoder.identifying_keys())

    def object_hook(self, object_dict):

        if any(key in self._evqe_population_decoder.identifying_keys() for key in object_dict.keys()):
            return self._evqe_population_decoder.object_hook(object_dict)

        if "complex_number_real_value" in object_dict or "complex_number_imaginary_value" in object_dict:
            return self.parse_complex_number(object_dict)

        if (
            "quasidistribution_data" in object_dict
            or "quasidistribution_shots" in object_dict
            or "quasidistribution_stdev_bound" in object_dict
        ):
            return self.parse_quasidistribution(object_dict)

        if "qiskit_quantum_circuit" in object_dict:
            return self.parse_quantum_circuit(object_dict)

        if (
            "base_population_evaluation_population" in object_dict
            or "base_population_evaluation_expectation_values" in object_dict
            or "base_population_evaluation_best_individual" in object_dict
            or "base_population_evaluation_best_expectation_value" in object_dict
        ):
            return self.parse_base_population_evaluation(object_dict)

        if any(
            key
            in [
                "evolving_ansatz_result_eigenvalue",
                "evolving_ansatz_result_aux_operators_evaluated",
                "evolving_ansatz_result_eigenstate",
                "evolving_ansatz_result_best_individual",
                "evolving_ansatz_result_circuit_evaluations",
                "evolving_ansatz_result_generations",
                "evolving_ansatz_population_evaluation_results",
                "evolving_ansatz_population_initial_state_circuit",
            ]
            for key in object_dict.keys()
        ):
            return self.parse_evolving_ansatz_result(object_dict)

    @staticmethod
    def parse_complex_number(object_dict):
        return complex(
            real=object_dict["complex_number_real_value"], imag=object_dict["complex_number_imaginary_value"]
        )

    @staticmethod
    def parse_quasidistribution(object_dict):
        return QuasiDistribution(
            data=dict(object_dict["quasidistribution_data"]),
            shots=object_dict["quasidistribution_shots"],
            stddev_upper_bound=object_dict["quasidistribution_stdev_bound"],
        )

    @staticmethod
    def parse_quantum_circuit(object_dict):
        str_encoded = object_dict["qiskit_quantum_circuit"]
        buffer = BytesIO(b64decode(str_encoded))
        return qpy_load(buffer)[0]

    @staticmethod
    def parse_base_population_evaluation(object_dict):
        return BasePopulationEvaluationResult(
            population=object_dict["base_population_evaluation_population"],
            expectation_values=tuple(object_dict["base_population_evaluation_expectation_values"]),
            best_individual=object_dict["base_population_evaluation_best_individual"],
            best_expectation_value=object_dict["base_population_evaluation_best_expectation_value"],
        )

    @staticmethod
    def parse_evolving_ansatz_result(object_dict):
        result = EvolvingAnsatzMinimumEigensolverResult()
        result.eigenvalue = object_dict["evolving_ansatz_result_eigenvalue"]

        aux_operators_evaluated = object_dict["evolving_ansatz_result_aux_operators_evaluated"]
        if isinstance(aux_operators_evaluated, dict):
            if aux_operators_evaluated["type"] == "list":
                aux_operators_evaluated = aux_operators_evaluated["values"]
            elif aux_operators_evaluated["type"] == "dict":
                aux_operators_evaluated = dict(aux_operators_evaluated["values"])
            else:
                aux_operators_evaluated = None
        else:
            aux_operators_evaluated = None

        result.aux_operators_evaluated = aux_operators_evaluated
        result.eigenstate = object_dict["evolving_ansatz_result_eigenstate"]
        result.best_individual = object_dict["evolving_ansatz_result_best_individual"]
        result.circuit_evaluations = object_dict["evolving_ansatz_result_circuit_evaluations"]
        result.generation = object_dict["evolving_ansatz_result_generations"]
        result.population_evaluation_results = object_dict["evolving_ansatz_population_evaluation_results"]
        result.initial_state_circuit = object_dict["evolving_ansatz_population_initial_state_circuit"]

        return result
