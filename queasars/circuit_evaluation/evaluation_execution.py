# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from threading import local, Lock
from time import sleep
from typing import Optional

from dask.distributed import Client
from qiskit.circuit import QuantumCircuit

from queasars.circuit_evaluation.circuit_evaluation import BaseCircuitEvaluator


class BaseCircuitEvaluationExecutor(ABC):

    @abstractmethod
    def evaluate_circuits(
        self, circuits: list[QuantumCircuit], parameter_values: list[list[float]], evaluator: BaseCircuitEvaluator
    ):
        pass


class BatchedMutexCircuitEvaluationExecutor(BaseCircuitEvaluationExecutor):

    def __init__(self, waiting_duration: Optional[float]):
        self.waiting_duration: Optional[float] = waiting_duration

        self._entry_lock = Lock()
        self._variable_lock = Lock()
        self._thread_counter: int = 0
        self._circuits: list[QuantumCircuit] = []
        self._parameter_values: list[list[float]] = []
        self._results: list[float] = []

    def evaluate_circuits(
        self, circuits: list[QuantumCircuit], parameter_values: list[list[float]], evaluator: BaseCircuitEvaluator
    ):
        with self._entry_lock:
            with self._variable_lock:
                thread_local = local()
                thread_local.circuit_list_index = len(self._circuits)
                self._circuits.extend(circuits)
                self._parameter_values.extend(parameter_values)
                thread_local.thread_index = self._thread_counter
                self._thread_counter += 1

        if self.waiting_duration is not None:
            sleep(self.waiting_duration)

        self._entry_lock.acquire(blocking=True)
        with self._variable_lock:
            if thread_local.thread_index == self._thread_counter - 1:
                thread_local.executor = True
                self._results = evaluator.evaluate_circuits(
                    circuits=self._circuits, parameter_values=self._parameter_values
                )
            else:
                thread_local.executor = False
                self._entry_lock.release()

        while len(self._results) <= 0:
            sleep(0.1)

        with self._variable_lock:
            thread_local.results = self._results[
                thread_local.circuit_list_index : thread_local.circuit_list_index + len(circuits)
            ]
            self._thread_counter -= 1

        if thread_local.executor:
            while self._thread_counter > 0:
                sleep(0.1)
            with self._variable_lock:
                self._results = []
                self._circuits = []
                self._parameter_values = []
            self._entry_lock.release()

        return thread_local.results


class DaskCircuitEvaluationExecutor(BaseCircuitEvaluationExecutor):

    def __init__(self, evaluator: BaseCircuitEvaluator, dask_client: Optional[Client]):
        self.evaluator: BaseCircuitEvaluator = evaluator
        self.dask_client: Optional[Client] = dask_client

    def evaluate_circuits(
        self, circuits: list[QuantumCircuit], parameter_values: list[list[float]], evaluator: BaseCircuitEvaluator
    ):
        pass
