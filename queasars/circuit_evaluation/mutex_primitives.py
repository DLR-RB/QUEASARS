# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from threading import Lock, Condition
from time import sleep
from typing import Callable, Optional, Generic, TypeVar, Any, Sequence

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import SamplerResult, EstimatorResult
from qiskit.primitives.base import BaseSampler, BaseEstimator
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import SparsePauliOp


T = TypeVar("T")


class BatchingMutexPrimitiveJobRunner(Generic[T]):
    """
    Class which wraps a method f(a, ..., z) which returns a PrimitiveJob[T] where a to z are Sequences of equal length.
    It batches the calls to f and enforces mutually exclusive access on f. Therefore, if multiple threads enter
    during the waiting time their arguments are concatenated: f(a_1 + ... +  a_n, ..., z_1 + ... +  z_n) and only
    one thread executes f, with the rest waiting to gather the results once they are available

    :param f: Callable which takes n sequences of equal length as input and
        returns a qiskit PrimitiveJob as a result
    :type f: Callable[[Sequence[Sequence[Any]], PrimitiveJob[T]]
    :param f_n_args: fixed amount of arguments which must be provided to f
    :type f_n_args: int
    :param batch_waiting_duration: Amount of time the last entering thread waits for others to enter after itself.
        Specifying no waiting duration might lead very small batches.
    :type batch_waiting_duration: Optional[float]
    """

    def __init__(
        self,
        f: Callable[[Sequence[Sequence[Any]]], PrimitiveJob[T]],
        f_n_args: int,
        batch_waiting_duration: Optional[float],
    ):
        """
        Constructor Method
        """
        self.f: Callable[[Sequence[Sequence[Any]]], PrimitiveJob[T]] = f
        self.batch_waiting_duration: Optional[float] = batch_waiting_duration
        self.n_args: int = f_n_args

        self._entry_lock = Lock()
        self._variable_lock = Lock()
        self._internal_wait_condition = Condition()
        self._external_wait_condition = Condition()
        self._thread_counter: int = 0
        self._entry_counter: int = 0
        self._batched_args: tuple[tuple[Any, ...], ...] = tuple()
        self._batch_length: int = 0
        self._result: Optional[T] = None
        self._exception: Optional[Exception] = None

    def run(self, *args: Sequence[Any]) -> tuple[T, int]:
        """
        Runs f for the batched arguments provided in args by this and other threads.
        Returns the result for the whole batch the index at which the given args were concatenated
        to the batch.

        :arg args: arguments for f provided by this thread
        :type args: Sequence[Sequence[Any]]
        :return: A tuple containing the result for the whole batch and the index
            at which the given args were concatenated
        :rtype: tuple[T, int]
        """

        # Check that the amount of arguments matches the amount specified in the constructor
        if len(args) != self.n_args:
            raise ValueError(f"The amount of arguments is {len(args)} but should be {self.n_args}!")

        # Check that all arguments are of the same length
        arg_length = len(args[0])
        for arg in args:
            if len(arg) != arg_length:
                raise ValueError("All arguments must be sequences of the same length!")

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

                    # Remember the index at which the thread's arguments are appended to the batch.
                    batch_index = self._batch_length
                    # Append the thread's arguments to the batch.
                    if self._batch_length == 0:
                        self._batched_args = tuple((*arg,) for arg in args)
                        self._batch_length = arg_length
                    else:
                        self._batched_args = tuple((*previous, *new) for previous, new in zip(self._batched_args, args))
                        self._batch_length = self._batch_length + arg_length

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
                self._result = self.f(*self._batched_args).result()
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
                self._batched_args = tuple()
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


class BatchingMutexSampler(BaseSampler[PrimitiveJob[SamplerResult]]):
    """
    Wrapper class for qiskit Sampler primitives which makes them threadsafe by batching concurrent requests and
    realizing mutual exclusion. Due to the batching nature of the BatchingMutexSampler it does not support
    run options in the run method. Any configuration needs to be done on the Sampler before it is wrapped.

    :param sampler: Sampler primitive to wrap
    :type sampler: BaseSampler
    :param waiting_duration: time in seconds to wait after the most recent request until starting the batch.
        No waiting time may result in very small batches
    :type waiting_duration: Optional[float]
    """

    def __init__(self, sampler: BaseSampler, waiting_duration: Optional[float]):
        """Constructor Method"""
        super().__init__()
        self.sampler: BaseSampler = sampler
        self.waiting_duration: Optional[float] = waiting_duration
        self._runner: BatchingMutexPrimitiveJobRunner[SamplerResult] = BatchingMutexPrimitiveJobRunner(
            f=self.sampler.run, batch_waiting_duration=waiting_duration, f_n_args=2
        )

    def _call(self, circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...]):
        total_sampling_result, batch_index = self._runner.run(circuits, parameter_values)
        input_length: int = len(circuits)
        private_result = SamplerResult(
            quasi_dists=total_sampling_result.quasi_dists[batch_index : batch_index + input_length],
            metadata=total_sampling_result.metadata[batch_index : batch_index + input_length],
        )
        return private_result

    def _run(
        self, circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...], **run_options
    ) -> PrimitiveJob[SamplerResult]:
        if len(run_options) != 0:
            raise ValueError(
                "The BatchingMutexSampler does not support run options due it's batching nature.\n"
                + "Please set any options before wrapping the Sampler!"
            )
        job = PrimitiveJob(self._call, circuits, parameter_values)
        job.submit()
        return job


class BatchingMutexEstimator(BaseEstimator[PrimitiveJob[EstimatorResult]]):
    """
    Wrapper class for qiskit Estimator primitives which makes them threadsafe by batching concurrent requests and
    realizing mutual exclusion. Due to the batching nature of the BatchingMutexEstimator it does not support
    run options in the run method. Any configuration needs to be done on the Estimator before it is wrapped.

    :param estimator: Estimator primitive to wrap
    :type estimator: BaseEstimator
    :param waiting_duration: time in seconds to wait after the most recent request until starting the batch.
        No waiting time may result in very small batches
    :type waiting_duration: Optional[float]
    """

    def __init__(self, estimator: BaseEstimator, waiting_duration: Optional[float]):
        """Constructor Method"""
        super().__init__()
        self.estimator: BaseEstimator = estimator
        self.waiting_duration: Optional[float] = waiting_duration
        self._runner: BatchingMutexPrimitiveJobRunner[EstimatorResult] = BatchingMutexPrimitiveJobRunner(
            f=self.estimator.run, batch_waiting_duration=waiting_duration, f_n_args=3
        )

    def _call(
        self,
        circuits: tuple[QuantumCircuit],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
    ):
        total_sampling_result, batch_index = self._runner.run(circuits, observables, parameter_values)
        input_length: int = len(circuits)
        private_result = EstimatorResult(
            values=total_sampling_result.values[batch_index : batch_index + input_length],
            metadata=total_sampling_result.metadata[batch_index : batch_index + input_length],
        )
        return private_result

    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> PrimitiveJob[EstimatorResult]:
        if len(run_options) != 0:
            raise ValueError(
                "The BatchingMutexEstimator does not support run options due it's batching nature.\n"
                + "Please set any options before wrapping the Estimator!"
            )
        job = PrimitiveJob(self._call, circuits, observables, parameter_values)
        job.submit()
        return job
