# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
import logging
from threading import Lock, Condition, local
from time import sleep
from typing import Union, Callable, Optional
from concurrent.futures import Future, wait
import datetime

from dask.distributed import Client

from qiskit.primitives.base import BaseSampler, BaseEstimator
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives import SamplerResult, EstimatorResult
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


def _call_with_sampler(
    sampler: BaseSampler,
    circuits: tuple[QuantumCircuit, ...],
    parameter_values: tuple[tuple[float, ...], ...],
):
    return sampler.run(circuits=circuits, parameter_values=parameter_values).result()


def _call_with_sampler_callable(
    sampler_callable: Callable[[], BaseSampler],
    circuits: tuple[QuantumCircuit, ...],
    parameter_values: tuple[tuple[float, ...], ...],
):
    return sampler_callable().run(circuits=circuits, parameter_values=parameter_values).result()


class BatchingMutexSampler(BaseSampler[PrimitiveJob[SamplerResult]]):

    def __init__(self, sampler: BaseSampler, waiting_duration: Optional[float]):
        super().__init__()
        self.sampler: BaseSampler = sampler
        self.waiting_duration: Optional[float] = waiting_duration

        self._entry_lock = Lock()
        self._variable_lock = Lock()
        self._is_finished_condition = Condition()
        self._thread_counter: int = 0
        self._circuits: tuple[QuantumCircuit, ...] = tuple()
        self._parameter_values: tuple[tuple[float, ...], ...] = tuple()
        self._result: Optional[SamplerResult] = None

    def _call(self, circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...]):
        thread_local = local()
        thread_local.acquired_both_locks = False
        while not thread_local.acquired_both_locks:
            self._entry_lock.acquire(blocking=True)
            if self._variable_lock.acquire(blocking=True, timeout=0.1):
                thread_local.acquired_both_locks = True
                thread_local.circuit_list_index = len(self._circuits)
                self._circuits = (*self.circuits, *circuits)
                self._parameter_values = (*self._parameter_values, *parameter_values)
                thread_local.thread_index = self._thread_counter
                sleep(0.05)
                self._thread_counter = self._thread_counter + 1
                self._entry_lock.release()
                self._variable_lock.release()
            else:
                self._entry_lock.release()

        if self.waiting_duration is not None:
            sleep(self.waiting_duration)

        self._variable_lock.acquire()
        if thread_local.thread_index == self._thread_counter - 1:
            self._entry_lock.acquire(blocking=True)
            thread_local.executor = True
            self._result = self.sampler.run(circuits=self._circuits, parameter_values=self._parameter_values).result()
            self._variable_lock.release()
        else:
            thread_local.executor = False
            self._variable_lock.release()
            with self._is_finished_condition:
                self._is_finished_condition.wait()

        with self._variable_lock:
            try:
                if self._result is not None:
                    thread_local.quasi_dists = self._result.quasi_dists[
                        thread_local.circuit_list_index : thread_local.circuit_list_index + len(circuits)
                    ]
                    thread_local.meta_data = self._result.metadata[
                        thread_local.circuit_list_index : thread_local.circuit_list_index + len(circuits)
                    ]
                    thread_local.result = SamplerResult(
                        quasi_dists=thread_local.quasi_dists, metadata=thread_local.meta_data
                    )
                    self._thread_counter -= 1
                else:
                    raise Exception("Can't access result before it was made accessible!")
            except Exception as e:
                logging.warning(str(e))

        if thread_local.executor:
            thread_local.counter = 1000
            while not thread_local.counter < 1:
                with self._variable_lock:
                    thread_local.counter = self._thread_counter
                    with self._is_finished_condition:
                        self._is_finished_condition.notify_all()
                sleep(0.1)
            with self._variable_lock:
                self._result = None
                self._circuits = tuple()
                self._parameter_values = tuple()
                self._thread_counter = 0
            self._entry_lock.release()

        return thread_local.result

    def _run(
        self, circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...], **run_options
    ) -> PrimitiveJob[SamplerResult]:
        job = PrimitiveJob(self._call, circuits, parameter_values)
        job.submit()
        return job


class DaskDistributedSampler(BaseSampler[PrimitiveJob[SamplerResult]]):

    def __init__(self, sampler: Union[BaseSampler, Callable[[None], BaseSampler]], dask_client: Client):
        super().__init__()
        self.sampler: Union[BaseSampler, Callable[[], BaseSampler]] = sampler
        self.client: Client = dask_client

    def _call(
        self,
        circuits: tuple[QuantumCircuit, ...],
        parameter_values: tuple[tuple[float, ...], ...],
    ) -> SamplerResult:
        future: Future[SamplerResult]
        if isinstance(self.sampler, BaseSampler):
            future = self.client.submit(_call_with_sampler, self.sampler, circuits, parameter_values)
        elif callable(self.sampler):
            future = self.client.submit(_call_with_sampler_callable, self.sampler, circuits, parameter_values)
        else:
            raise ValueError("")
        return future.result()

    def _run(
        self, circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...], **run_options
    ) -> PrimitiveJob[SamplerResult]:
        job = PrimitiveJob(self._call, circuits, parameter_values)
        job.submit()
        return job


class BatchingMutexEstimator(BaseEstimator[PrimitiveJob[EstimatorResult]]):
    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob[EstimatorResult]:
        pass


class DaskDistributedEstimator(BaseEstimator[PrimitiveJob[EstimatorResult]]):

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob[EstimatorResult]:
        pass
