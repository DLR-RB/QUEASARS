# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from threading import Condition, Lock
from time import sleep
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar

from dask.utils import SerializableLock
from qiskit.primitives import (
    EstimatorPubLike,
    PrimitiveResult,
    PubResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.base import BaseEstimatorV2, BaseSamplerV2
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob, BasePrimitiveJob


PUBTYPE = TypeVar("PUBTYPE")
PUBRESULT = TypeVar("PUBRESULT")


class BatchingMutexPrimitiveJobRunner(Generic[PUBTYPE, PUBRESULT]):
    """
    Class which wraps a method f of the type Callable[[list[PUBTYPE]], PrimitiveJob[PrimitiveResult[PUBRESULT]]]
    It batches the calls to f and enforces mutually exclusive access on f. Therefore, if multiple threads enter
    during the waiting time their arguments are concatenated and only one thread executes f,
    with the rest waiting to gather the results once they are available

    :param f: Callable which takes a list of values as an input and returns a BasePrimitiveJob that processes and
                returns values for each input value.
    :type f: Callable[[list[PUBTYPE]], BasePrimitiveJob[PrimitiveResult[PUBRESULT], Any]]
    :param batch_waiting_duration: Amount of time the last entering thread waits for others to enter after itself.
        Specifying no waiting duration might lead very small batches.
    :type batch_waiting_duration: Optional[float]
    """

    def __init__(
        self,
        f: Callable[[list[PUBTYPE]], BasePrimitiveJob[PrimitiveResult[PUBRESULT], Any]],
        batch_waiting_duration: Optional[float],
    ):
        """
        Constructor Method
        """
        self.f: Callable[[list[PUBTYPE]], BasePrimitiveJob[PrimitiveResult[PUBRESULT], Any]] = f
        self.batch_waiting_duration: Optional[float] = batch_waiting_duration

        # The entry lock restricts whether a thread may add its circuit evaluation requests to the batch.
        # This is only allowed, if no other thread is adding its circuits currently and the batch
        # is not currently being processed.
        self._entry_lock: Lock = Lock()
        # The variable locks is used to protect shared variables which multiple threads may write to.
        # If a thread wants to write to a shared variable it must first acquire this lock.
        self._variable_lock: Lock = Lock()
        self._internal_wait_condition: Condition = Condition()
        self._external_wait_condition: Condition = Condition()
        self._thread_counter: int = 0
        self._entry_counter: int = 0
        self._batched_pubs: list[PUBTYPE] = []
        self._batch_length: int = 0
        self._result: Optional[PrimitiveResult[PUBRESULT]] = None
        self._exception: Optional[Exception] = None

    def run(self, pubs: list[PUBTYPE]) -> tuple[PrimitiveResult[PUBRESULT], int]:
        """
        Runs f for the batched arguments provided in args by this and other threads.
        Returns the result for the whole batch the index at which the given args were concatenated
        to the batch.

        :arg pubs: list of values to be processed
        :type pubs: list[PUBTYPE]
        :return: The tuple of the results for all batched calculations in a PrimitiveResult and the index
                    at which the results for this specific input data starts. To be exact, the results for this
                    calculation are at the indices start_index: start_index + len(pubs), where start_index is the
                    index returned by this method.
        :rtype: tuple[PrimitiveResult[PUBRESULT], int]:
        """

        # This region ensures that no thread may enter if a batched execution of f is currently running.
        # If f is not currently running it ensures that entering threads append their arguments to the batch
        # in an orderly and threadsafe manner.
        # To successfully enter this region, entry must be currently allowed (self._entry_lock must be available)
        # and this thread must be given write access to the shared memory (self._variable_lock must be acquired).
        acquired_both_locks: bool = False
        while not acquired_both_locks:

            # Try to acquire both needed locks.
            if self._entry_lock.acquire(blocking=True):
                if self._variable_lock.acquire(blocking=False):
                    # Both needed locks are acquired, the thread may continue.
                    acquired_both_locks = True

                    # Remember the index at which the thread's pubs are appended to the batch.
                    batch_index = self._batch_length
                    # Append the thread's pubs to the batch.
                    self._batched_pubs.extend(pubs)
                    self._batch_length = self._batch_length + len(pubs)

                    # Increase the counter to keep note of how many threads have entered so far.
                    self._thread_counter = self._thread_counter + 1

                    # The thread is now done entering. Release the locks and notify the other waiting threads that
                    # entry may be available again.
                    self._entry_lock.release()
                    self._variable_lock.release()
                    with self._external_wait_condition:
                        self._external_wait_condition.notify_all()
                else:
                    # Release the acquired lock, if only one could be acquired!
                    self._entry_lock.release()
            # If acquiring both locks was unsuccessful, wait to be notified of entry availability.
            if not acquired_both_locks:
                with self._external_wait_condition:
                    self._external_wait_condition.wait()

        # Wait for the given duration so that other threads may have time to enter.
        if self.batch_waiting_duration is not None:
            sleep(self.batch_waiting_duration)

        # Here the region begins in which the f is executed and the results are gathered by all threads.
        # Write access to shared memory is needed. Therefore, acquire self._variable_lock.
        self._variable_lock.acquire()

        # Check whether the thread arriving here is the last thread entering here.
        self._entry_counter = self._entry_counter + 1
        if self._entry_counter == self._thread_counter:

            # If it is the last thread, lock down entry by acquiring self._entry_lock.
            self._entry_lock.acquire(blocking=True)

            # Mark the thread as the executor and run the f.
            executor = True
            try:
                self._result = self.f(self._batched_pubs).result()
            except Exception as e:
                self._result = None
                self._exception = e

            # Write access is finished here, therefore release self._variable lock.
            self._variable_lock.release()

        else:
            # If it is not the last thread, mark it as not executing the f.
            executor = False
            # Write access is finished here, therefore release self._variable lock.
            self._variable_lock.release()
            # Wait to be notified that the results of executing the f are available.
            with self._internal_wait_condition:
                self._internal_wait_condition.wait()

        # Acquire the _variable_lock to gain write access.
        with self._variable_lock:
            # Try to gather the batch results.
            if self._result is not None:
                result = self._result
                # Since non-executing threads are finished here, mark their exiting by reducing the _thread_counter.
                self._thread_counter -= 1
                # Notify a subsequent thread of the availability of the results.
                with self._internal_wait_condition:
                    self._internal_wait_condition.notify()
            else:
                # Since non-executing threads are finished here, mark their exiting by reducing the _thread_counter.
                self._thread_counter -= 1
                with self._internal_wait_condition:
                    self._internal_wait_condition.notify()
                if self._exception is None:
                    raise ValueError("Result was not yet ready to retrieve!")
                raise self._exception

        if executor:
            # The executing thread waits for all other threads to exit.
            while self._thread_counter > 0:
                with self._internal_wait_condition:
                    self._internal_wait_condition.wait(0.5)
                with self._internal_wait_condition:
                    self._internal_wait_condition.notify()

            # After all threads have gathered their results, gain write access by acquiring _variable_lock,
            # then reset the shared variables to prepare the next batch.
            with self._variable_lock:
                self._result = None
                self._batched_pubs = []
                self._batch_length = 0
                self._thread_counter = 0
                self._entry_counter = 0
                if self._exception is not None:
                    exception = self._exception
                    self._exception = None
                    raise exception

            # Finally re-enable entry and notify the waiting threads so that next batch may be gathered.
            self._entry_lock.release()
            with self._external_wait_condition:
                self._external_wait_condition.notify_all()

        return result, batch_index


class MutexSampler(BaseSamplerV2):
    """
    Wrapper class for qiskit SamplerV2 primitives which makes them threadsafe by realizing a simple mutually
    exclusive access on the sampler. Due to the use of a dask SerializableLock, this mutually exclusive access
    holds over different dask worker threads in the same process.

    :param sampler: SamplerV2 primitive which shall be wrapped for mutually exclusive access
    :type sampler: BaseSamplerV2
    """

    def __init__(self, sampler: BaseSamplerV2):
        super().__init__()
        self._sampler: BaseSamplerV2 = sampler
        self._lock: SerializableLock = SerializableLock()

    def run(
        self, pubs: Iterable[SamplerPubLike], *args, shots: Optional[int] = None
    ) -> BasePrimitiveJob[PrimitiveResult[SamplerPubResult], Any]:
        with self._lock:
            return self._sampler.run(pubs=pubs, *args, shots=shots)


class BatchingMutexSampler(BaseSamplerV2):
    """
    Wrapper class for qiskit SamplerV2 primitives which makes them threadsafe by batching concurrent requests and
    realizing mutual exclusion.

    :param sampler: SamplerV2 primitive to wrap
    :type sampler: BaseSamplerV2
    :param waiting_duration: time in seconds to wait after the most recent request until starting the batch.
        No waiting time may result in very small batches
    :type waiting_duration: Optional[float]
    """

    def __init__(self, sampler: BaseSamplerV2, waiting_duration: Optional[float]):
        """Constructor Method"""
        super().__init__()
        self._sampler: BaseSamplerV2 = sampler
        self.waiting_duration: Optional[float] = waiting_duration
        self._runner: BatchingMutexPrimitiveJobRunner[SamplerPub, SamplerPubResult] = BatchingMutexPrimitiveJobRunner(
            f=self._sample, batch_waiting_duration=waiting_duration
        )

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: Optional[int] = None
    ) -> BasePrimitiveJob[PrimitiveResult[SamplerPubResult], Any]:
        coerced_pubs: list[SamplerPub] = [SamplerPub.coerce(pub, shots) for pub in pubs]
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        n_pubs = len(pubs)
        result, start_index = self._runner.run(pubs=pubs)
        selected_results: list[SamplerPubResult] = [result[i] for i in range(start_index, start_index + n_pubs)]
        return PrimitiveResult(
            pub_results=selected_results,
            metadata=result.metadata,
        )

    def _sample(self, pubs: list[SamplerPub]) -> BasePrimitiveJob[PrimitiveResult[SamplerPubResult], Any]:
        return self._sampler.run(pubs=pubs)


class MutexEstimator(BaseEstimatorV2):
    """
    Wrapper class for qiskit EstimatorV2 primitives which makes them threadsafe by realizing a simple mutually
    exclusive access on the estimator. Due to the use of a dask SerializableLock, this mutually exclusive access
    holds over different dask worker threads in the same process.

    :param estimator: estimator primitive which shall be wrapped for mutually exclusive access
    :type estimator: BaseEstimatorV2
    """

    def __init__(self, estimator: BaseEstimatorV2):
        super().__init__()
        self._estimator: BaseEstimatorV2 = estimator
        self._lock: SerializableLock = SerializableLock()

    def run(
        self, pubs: Iterable[EstimatorPubLike], *args, precision: Optional[float] = None
    ) -> BasePrimitiveJob[PrimitiveResult[PubResult], Any]:
        with self._lock:
            return self._estimator.run(pubs=pubs, *args, precision=precision)


class BatchingMutexEstimator(BaseEstimatorV2):
    """
    Wrapper class for qiskit EstimatorV2 primitives which makes them threadsafe by batching concurrent requests and
    realizing mutual exclusion.

    :param estimator: EstimatorV2 primitive to wrap
    :type estimator: BaseEstimatorV2
    :param waiting_duration: time in seconds to wait after the most recent request until starting the batch.
        No waiting time may result in very small batches
    :type waiting_duration: Optional[float]
    """

    def __init__(self, estimator: BaseEstimatorV2, waiting_duration: Optional[float]):
        """Constructor Method"""
        super().__init__()
        self._estimator: BaseEstimatorV2 = estimator
        self.waiting_duration: Optional[float] = waiting_duration
        self._runner: BatchingMutexPrimitiveJobRunner[EstimatorPub, PubResult] = BatchingMutexPrimitiveJobRunner(
            f=self._estimate, batch_waiting_duration=waiting_duration
        )

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: Optional[float] = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        coerced_pubs: list[EstimatorPub] = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        n_pubs = len(pubs)
        result, start_index = self._runner.run(pubs=pubs)
        selected_results: list[PubResult] = [result[i] for i in range(start_index, start_index + n_pubs)]
        return PrimitiveResult(
            pub_results=selected_results,
            metadata=result.metadata,
        )

    def _estimate(self, pubs: list[EstimatorPub]) -> BasePrimitiveJob[PrimitiveResult[PubResult], Any]:
        return self._estimator.run(pubs=pubs)
