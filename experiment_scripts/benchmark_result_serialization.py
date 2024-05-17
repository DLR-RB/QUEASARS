# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass
from json import JSONEncoder, JSONDecoder
from typing import Any

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
    n_qubits: int
    instance_nr: int
    seed: int
    expectation_evaluation_counts: list[float]
    expectation_values: list[float]
    min_energy: float
    max_energy: float
    measurement_distribution: QuasiDistribution
    state_translations: dict[int, JobShopSchedulingResult]


@dataclass
class EVQEBenchmarkResult:
    n_qubits: int
    instance_nr: int
    seed: int
    result: EvolvingAnsatzMinimumEigensolverResult
    min_energy: float
    max_energy: float
    state_translations: dict[int, JobShopSchedulingResult]


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
            return {
                "vqaresult_n_qubits": o.n_qubits,
                "vqaresult_instance_nr": o.instance_nr,
                "vqaresult_seed": o.seed,
                "vqaresult_evaluation_counts": o.expectation_evaluation_counts,
                "vqaresult_expectation_values": o.expectation_values,
                "vqaresult_min_energy": o.min_energy,
                "vqaresult_max_energy": o.max_energy,
                "vqaresult_measurement_distribution": self.default(o.measurement_distribution),
                "vqaresult_state_translations": [
                    [key, self.default(value)] for key, value in o.state_translations.items()
                ],
            }

        if isinstance(o, EVQEBenchmarkResult):
            return {
                "evqeresult_n_qubits": o.n_qubits,
                "evqeresult_instance_nr": o.instance_nr,
                "evqeresult_seed": o.seed,
                "evqeresult_result": self.default(o.result),
                "evqeresult_min_energy": o.min_energy,
                "evqeresult_max_energy": o.max_energy,
                "evqeresult_state_translations": [
                    [key, self.default(value)] for key, value in o.state_translations.items()
                ],
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
            ]
            for key in object_dict.keys()
        ):
            return self.parse_evqeresult(object_dict)

    @staticmethod
    def parse_vqaresult(object_dict):
        return VQABenchmarkResult(
            n_qubits=object_dict["vqaresult_n_qubits"],
            instance_nr=object_dict["vqaresult_instance_nr"],
            seed=object_dict["vqaresult_seed"],
            expectation_evaluation_counts=object_dict["vqaresult_evaluation_counts"],
            expectation_values=object_dict["vqaresult_expectation_values"],
            min_energy=object_dict["vqaresult_min_energy"],
            max_energy=object_dict["vqaresult_max_energy"],
            measurement_distribution=object_dict["vqaresult_measurement_distribution"],
            state_translations=object_dict["vqaresult_state_translations"],
        )

    @staticmethod
    def parse_evqeresult(object_dict):
        return EVQEBenchmarkResult(
            n_qubits=object_dict["evqeresult_n_qubits"],
            instance_nr=object_dict["evqeresult_instance_nr"],
            seed=object_dict["evqeresult_seed"],
            result=object_dict["evqeresult_result"],
            min_energy=object_dict["evqeresult_min_energy"],
            max_energy=object_dict["evqeresult_max_energy"],
            state_translations=object_dict["evqeresult_state_translations"],
        )
