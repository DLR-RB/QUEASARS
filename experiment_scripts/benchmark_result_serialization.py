# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass
from json import JSONEncoder, JSONDecoder
from typing import Any, Optional

from qiskit.result import QuasiDistribution

from queasars.job_shop_scheduling.problem_instances import JobShopSchedulingResult
from queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver_result import (
    EvolvingAnsatzMinimumEigensolverResult,
)
from queasars.job_shop_scheduling.serialization import JSSPJSONEncoder, JSSPJSONDecoder
from queasars.minimum_eigensolvers.base.serialization import (
    EvolvingAnsatzMinimumEigensolverResultJSONEncoder,
    EvolvingAnsatzMinimumEigensolverResultJSONDecoder,
)


@dataclass
class VQABenchmarkResult:
    n_qubits: Optional[int] = None
    instance_nr: Optional[int] = None
    seed: Optional[int] = None
    best_expectation_value: Optional[float] = None
    best_parameter_values: Optional[list[float]] = None
    expectation_evaluation_counts: Optional[list[float]] = None
    expectation_values: Optional[list[float]] = None
    measurement_distribution: Optional[QuasiDistribution] = None
    state_translations: Optional[dict[int, JobShopSchedulingResult]] = None
    optimal_makespan: Optional[int] = None
    min_penalty: Optional[float] = None
    min_energy: Optional[float] = None
    min_subopt_energy: Optional[float] = None
    max_opt_energy: Optional[float] = None
    max_energy: Optional[float] = None


@dataclass
class EVQEBenchmarkResult:
    n_qubits: Optional[int] = None
    instance_nr: Optional[int] = None
    seed: Optional[int] = None
    result: Optional[EvolvingAnsatzMinimumEigensolverResult] = None
    state_translations: Optional[dict[int, JobShopSchedulingResult]] = None
    optimal_makespan: Optional[int] = None
    min_penalty: Optional[float] = None
    min_energy: Optional[float] = None
    min_subopt_energy: Optional[float] = None
    max_opt_energy: Optional[float] = None
    max_energy: Optional[float] = None


class ResultEncoder(JSONEncoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evolving_ansatz_encoder = EvolvingAnsatzMinimumEigensolverResultJSONEncoder(*args, **kwargs)
        self._jssp_encoder = JSSPJSONEncoder(*args, **kwargs)

    def default(self, o: Any):

        if any(isinstance(o, t) for t in EvolvingAnsatzMinimumEigensolverResultJSONEncoder.serializable_types()):
            return self._evolving_ansatz_encoder.default(o)

        if any(isinstance(o, t) for t in JSSPJSONEncoder.serializable_types()):
            return self._jssp_encoder.default(o)

        if isinstance(o, VQABenchmarkResult):

            if o.state_translations is None:
                state_translations = None
            else:
                state_translations = [[key, self.default(value)] for key, value in o.state_translations.items()]

            return {
                "vqaresult_n_qubits": o.n_qubits,
                "vqaresult_instance_nr": o.instance_nr,
                "vqaresult_seed": o.seed,
                "vqaresult_evaluation_counts": o.expectation_evaluation_counts,
                "vqaresult_expectation_values": o.expectation_values,
                "vqaresult_min_energy": o.min_energy,
                "vqaresult_max_energy": o.max_energy,
                "vqaresult_measurement_distribution": self.default(o.measurement_distribution),
                "vqaresult_state_translations": state_translations,
                "vqaresult_optimal_makespan": o.optimal_makespan,
                "vqaresult_max_opt_energy": o.max_opt_energy,
                "vqaresult_best_expectation_value": o.best_expectation_value,
                "vqaresult_best_parameter_values": o.best_parameter_values,
                "vqaresult_min_penalty": o.min_penalty,
                "vqaresult_min_subopt_energy": o.min_subopt_energy,
            }

        if isinstance(o, EVQEBenchmarkResult):

            if o.state_translations is None:
                state_translations = None
            else:
                state_translations = [[key, self.default(value)] for key, value in o.state_translations.items()]

            return {
                "evqeresult_n_qubits": o.n_qubits,
                "evqeresult_instance_nr": o.instance_nr,
                "evqeresult_seed": o.seed,
                "evqeresult_result": self.default(o.result),
                "evqeresult_min_energy": o.min_energy,
                "evqeresult_max_energy": o.max_energy,
                "evqeresult_state_translations": state_translations,
                "evqeresult_optimal_makespan": o.optimal_makespan,
                "evqeresult_max_opt_energy": o.max_opt_energy,
                "evqeresult_min_penalty": o.min_penalty,
                "evqeresult_min_subopt_energy": o.min_subopt_energy,
            }


class ResultDecoder(JSONDecoder):

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
        self._evolving_ansatz_decoder = EvolvingAnsatzMinimumEigensolverResultJSONDecoder(*args, **kwargs)
        self._jssp_decoder = JSSPJSONDecoder(*args, **kwargs)

    def object_hook(self, object_dict: dict):

        if any(
            key in EvolvingAnsatzMinimumEigensolverResultJSONDecoder.identifying_keys() for key in object_dict.keys()
        ):
            return self._evolving_ansatz_decoder.object_hook(object_dict)

        if any(key in JSSPJSONDecoder.identifying_keys() for key in object_dict.keys()):
            return self._jssp_decoder.object_hook(object_dict)

        if any(
            key
            in [
                "vqaresult_n_qubits",
                "vqaresult_instance_nr",
                "vqaresult_seed",
                "vqaresult_evaluation_counts",
                "vqaresult_expectation_values",
                "vqaresult_min_energy",
                "vqaresult_max_energy",
                "vqaresult_measurement_distribution",
                "vqaresult_state_translations",
                "vqaresult_optimal_makespan",
                "vqaresult_max_opt_energy",
                "vqaresult_best_expectation_value",
                "vqaresult_best_parameter_values",
                "vqaresult_min_penalty",
                "vqaresult_min_subopt_energy",
            ]
            for key in object_dict.keys()
        ):
            return self.parse_vqaresult(object_dict)

        if any(
            key
            in [
                "evqeresult_n_qubits",
                "evqeresult_instance_nr",
                "evqeresult_seed",
                "evqeresult_result",
                "evqeresult_min_energy",
                "evqeresult_max_energy",
                "evqeresult_state_translations",
                "evqeresult_optimal_makespan",
                "evqeresult_max_opt_energy",
                "evqeresult_min_penalty",
                "evqeresult_min_subopt_energy",
            ]
            for key in object_dict.keys()
        ):
            return self.parse_evqeresult(object_dict)

    @staticmethod
    def parse_vqaresult(object_dict):

        state_translations = object_dict.get("vqaresult_state_translations")
        if state_translations is not None:
            state_translations = dict(state_translations)

        return VQABenchmarkResult(
            n_qubits=object_dict.get("vqaresult_n_qubits"),
            instance_nr=object_dict.get("vqaresult_instance_nr"),
            seed=object_dict.get("vqaresult_seed"),
            expectation_evaluation_counts=object_dict.get("vqaresult_evaluation_counts"),
            expectation_values=object_dict.get("vqaresult_expectation_values"),
            min_energy=object_dict.get("vqaresult_min_energy"),
            max_energy=object_dict.get("vqaresult_max_energy"),
            measurement_distribution=object_dict.get("vqaresult_measurement_distribution"),
            state_translations=state_translations,
            optimal_makespan=object_dict.get("vqaresult_optimal_makespan"),
            max_opt_energy=object_dict["vqaresult_max_opt_energy"],
            best_expectation_value=object_dict.get("vqaresult_best_expectation_value"),
            best_parameter_values=object_dict.get("vqaresult_best_parameter_values"),
            min_penalty=object_dict.get("vqaresult_min_penalty"),
            min_subopt_energy=object_dict.get("vqaresult_min_subopt_energy"),
        )

    @staticmethod
    def parse_evqeresult(object_dict):

        state_translations = object_dict.get("evqeresult_state_translations")
        if state_translations is not None:
            state_translations = dict(state_translations)

        return EVQEBenchmarkResult(
            n_qubits=object_dict.get("evqeresult_n_qubits"),
            instance_nr=object_dict.get("evqeresult_instance_nr"),
            seed=object_dict.get("evqeresult_seed"),
            result=object_dict.get("evqeresult_result"),
            min_energy=object_dict.get("evqeresult_min_energy"),
            max_energy=object_dict.get("evqeresult_max_energy"),
            state_translations=state_translations,
            optimal_makespan=object_dict.get("evqeresult_optimal_makespan"),
            max_opt_energy=object_dict.get("evqeresult_max_opt_energy"),
            min_penalty=object_dict.get("evqeresult_min_penalty"),
            min_subopt_energy=object_dict.get("evqeresult_min_subopt_energy"),
        )
