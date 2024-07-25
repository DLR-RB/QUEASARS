# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from typing import Any, Iterable, Optional, TypeVar, Union

from qiskit.primitives import (
    SamplerPubLike,
    BasePrimitiveJob,
    PrimitiveResult,
    SamplerPubResult,
    EstimatorPubLike,
    PubResult,
)
from qiskit.primitives.base import BaseSamplerV2, BaseEstimatorV2
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from qiskit.transpiler import PassManager


T = TypeVar("T")


class TranspilingSamplerV2(BaseSamplerV2):
    """
    Wrapper class for qiskit's SamplerV2 primitives, that transpiles the quantum circuits using a PassManager
    before handing them to the wrapped sampler.

    :param sampler: The SamplerV2 primitive to wrap.
    :type sampler: SamplerV2
    :param pass_manager: PassManager which specifies the transpilation procedure.
    :type pass_manager: PassManager
    """

    def __init__(self, sampler: BaseSamplerV2, pass_manager: PassManager):
        super().__init__()
        self._sampler = sampler
        self._pass_manager = pass_manager

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: Optional[int] = None
    ) -> BasePrimitiveJob[PrimitiveResult[SamplerPubResult], Any]:
        def _ensure_tuple(value: Union[T, tuple[T]]) -> tuple[T]:
            if isinstance(value, tuple):
                return value
            return (value,)

        pubs = (_ensure_tuple(pub) for pub in pubs)
        pubs = ((self._pass_manager.run(circuits=pub[0]), *pub[1:]) for pub in pubs)
        return self._sampler.run(pubs, shots=shots)


class TranspilingEstimatorV2(BaseEstimatorV2):
    """
    Wrapper class for qiskit's EstimatorV2 primitives, that transpiles the quantum circuits using a PassManager
    before handing them to the wrapped estimator.

    :param estimator: The EstimatorV2 primitive to wrap.
    :type estimator: SamplerV2
    :param pass_manager: PassManager which specifies the transpilation procedure.
    :type pass_manager: PassManager
    """

    def __init__(self, estimator: BaseEstimatorV2, pass_manager: PassManager):
        super().__init__()
        self._estimator = estimator
        self._pass_manager = pass_manager

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: Optional[float] = None
    ) -> BasePrimitiveJob[PrimitiveResult[PubResult], Any]:

        def apply_pass_manager(pub: EstimatorPubLike) -> EstimatorPubLike:
            if isinstance(pub, EstimatorPub):
                return EstimatorPub(
                    circuit=self._pass_manager.run(pub.circuit),
                    observables=pub.observables,
                    parameter_values=pub.parameter_values,
                    precision=pub.precision,
                    validate=False,
                )
            return self._pass_manager.run(circuits=pub[0]), *pub[1:]

        pubs = (apply_pass_manager(pub) for pub in pubs)
        return self._estimator.run(pubs, precision=precision)
