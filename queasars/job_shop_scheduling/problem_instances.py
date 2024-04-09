# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, cast
from typing_extensions import TypeGuard
from textwrap import indent


@dataclass(frozen=True)
class Machine:
    """Dataclass representing a machine in the context of the Job Shop Scheduling Problem

    :param name: identifier of the machine
    :type name: str
    """

    name: str

    def __post_init__(self):
        if self.name == "":
            raise JobShopSchedulingProblemException("The name of a Machine cannot be an empty string!")

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Operation:
    """Dataclass representing an operation in the context of the Job Shop Scheduling Problem. An operation is one
    step in finishing a Job

    :param name: identifier of the operation. Must not be an empty string.
        Must be unique between all operations of a job
    :type name: str
    :param job_name: name of the job to which this operation belongs. Must exactly match
        the name of the job it is part of
    :type job_name: str
    :param machine: machine on which this operation must be executed
    :type machine: Machine
    :param processing_duration: amount of time units needed to finish the operation. Must be at least 1
    :type processing_duration: int
    """

    name: str
    job_name: str
    machine: Machine
    processing_duration: int

    @property
    def identifier(self) -> str:
        """
        Unique operation identifier in a valid JobShopSchedulingProblemInstance, consisting of both the job
        and operation name.

        :return: a unique string identifier for the operation
        :rtype: str
        """
        return self.job_name + "_" + self.name

    def __post_init__(self):
        if self.name == "":
            raise JobShopSchedulingProblemException("The name of an Operation cannot be an empty string!")
        if self.job_name == "":
            raise JobShopSchedulingProblemException("The job_name of an Operation cannot be an empty string!")
        if self.processing_duration <= 0:
            raise JobShopSchedulingProblemException(
                f"The processing_duration of an Operation must at least be one, but it was {self.processing_duration}"
            )

    def __repr__(self):
        return f"{self.identifier}({self.machine.name}, {self.processing_duration})"


@dataclass(frozen=True)
class Job:
    """Dataclass representing a job in the context of the Job Shop Scheduling Problem

    :param name: identifier of the job
    :type name: str
    :param operations: tuple of the operations which need to be processed to finish this job.
        They need to be processed in the order specified by the given tuple. The tuple needs to contain at least
        one operation
    :type operations: tuple[Operation, ...]
    """

    name: str
    operations: tuple[Operation, ...]

    def is_consistent_with_machines(self, machines: tuple[Machine, ...]) -> bool:
        """Method which checks that the operations of a job visit only a limited collection of machines

        :arg machines: machines which the operations of this job may use
        :type machines: tuple[Machine, ...]

        :return: True if only the given machines are visited, False otherwise
        :rtype: bool
        """
        for operation in self.operations:
            if machines is not None and operation.machine not in machines:
                return False
        return True

    def __post_init__(self):
        if self.name == "":
            raise JobShopSchedulingProblemException("The name of a Job cannot be an empty string!")

        if len(self.operations) == 0:
            raise JobShopSchedulingProblemException(
                "This job contains no operations! A job must contain at least 1 operation!"
            )

        operation_identifiers = set(map(lambda x: x.identifier, self.operations))
        if len(operation_identifiers) != len(self.operations):
            raise JobShopSchedulingProblemException("The identifiers of all operations within a job must be unique!")

        visited_machines = set()
        for operation in self.operations:
            if operation.job_name != self.name:
                raise JobShopSchedulingProblemException(
                    f"The job_name of an operation was mismatched! Expected {self.name}, Got: {operation.job_name}"
                )
            if operation.machine in visited_machines:
                raise JobShopSchedulingProblemException(
                    f"The machine {operation.machine} was visited by more than one operation!"
                )
            visited_machines.add(operation.machine)

    def __repr__(self):
        header = f"{self.name}:\n"
        text = ""
        for operation in self.operations:
            text += str(operation) + "\n"
        return header + indent(
            text=text,
            prefix=" " * 2,
        )


@dataclass(frozen=True)
class JobShopSchedulingProblemInstance:
    """Dataclass representing a Job Shop Scheduling Problem

    :param name: identifier of the job shop scheduling problem instance
    :type name: str
    :param machines: tuple of all machines available for processing
    :type machines: tuple[Machine, ...]
    :param jobs: tuple of all jobs which need to be processed
    :type jobs: tuple[Job, ...]
    """

    name: str
    machines: tuple[Machine, ...]
    jobs: tuple[Job, ...]

    def __post_init__(self):
        if self.name == "":
            raise JobShopSchedulingProblemException(
                "The name of a JobShopSchedulingProblemInstance may not be an empty string!"
            )

        if len(set(self.machines)) != len(self.machines):
            raise JobShopSchedulingProblemException(
                "The Machines in a JobShopSchedulingProblemInstance must be unique!"
            )

        job_names = set(map(lambda x: x.name, self.jobs))
        if len(job_names) != len(self.jobs):
            raise JobShopSchedulingProblemException(
                "The names of the Jobs in a JobShopSchedulingProblemInstance must be unique!"
            )

        for job in self.jobs:
            if not job.is_consistent_with_machines(machines=self.machines):
                raise JobShopSchedulingProblemException(
                    "The Jobs in a JobShopSchedulingProblemInstance must not access "
                    + "other Machines than specified in its machines attribute!"
                )

    def __repr__(self):
        header = self.name + "\n"

        machine_header = "Machines:\n"
        machine_text = ""
        for machine in self.machines:
            machine_text += str(machine) + "\n"

        job_header = "Jobs:\n"
        job_text = ""
        for job in self.jobs:
            job_text += str(job)

        return (
            header
            + indent(text=machine_header, prefix=" " * 2)
            + indent(text=machine_text, prefix=" " * 4)
            + indent(text=job_header, prefix=" " * 2)
            + indent(text=job_text, prefix=" " * 4)
        )


@dataclass(frozen=True)
class PotentiallyScheduledOperation(ABC):
    """
    Abstract dataclass whose subclasses shall represent the scheduling status of an operation

    :param operation: operation whose scheduling status this object represents
    :type operation: Operation
    """

    operation: Operation

    @property
    @abstractmethod
    def is_scheduled(self) -> bool:
        """
        :return: whether the operation was successfully scheduled
        :rtype: bool
        """


@dataclass(frozen=True)
class UnscheduledOperation(PotentiallyScheduledOperation):
    """
    Dataclass which represents the fact that an operation was not successfully scheduled

    :param operation: operation which has not been scheduled
    :type operation: Operation
    """

    @property
    def is_scheduled(self) -> bool:
        return False

    def __repr__(self):
        return f"{str(self.operation)} was not scheduled."


@dataclass(frozen=True)
class ScheduledOperation(PotentiallyScheduledOperation):
    """
    Dataclass representing the fact that an Operation has been scheduled to start
    at a certain time

    :param operation: which has been scheduled
    :type operation: Operation
    :param start_time: time at which the operation has been scheduled to start
    :type start_time: int
    """

    start_time: int

    @property
    def is_scheduled(self) -> bool:
        return True

    @property
    def end_time(self) -> int:
        """
        :return: the end time of the operation, according to the schedule
        :rtype: int
        """
        return self.start_time + self.operation.processing_duration

    def __repr__(self):
        return f"{str(self.operation)} starts at: {self.start_time} and ends at: {self.end_time}"


def ensure_all_operations_are_scheduled(
    schedule: dict[Job, tuple[PotentiallyScheduledOperation, ...]]
) -> TypeGuard[dict[Job, tuple[ScheduledOperation, ...]]]:
    """Typeguard which checks that all operations in a schedule are actually scheduled

    :arg schedule: schedule to check
    :type schedule: dict[Job, tuple[PotentiallyScheduledOperation, ...]]
    :return: true if all operations are scheduled, false otherwise
    :rtype: bool
    """
    for _, job_schedule in schedule.items():
        if any(
            isinstance(potentially_scheduled_operation, UnscheduledOperation)
            for potentially_scheduled_operation in job_schedule
        ):
            return False
    return True


class JobShopSchedulingResult:
    """
    Class representing an attempted solution to the given job shop scheduling problem instance

    :param problem_instance: for which this schedule represents an attempted solution
    :type problem_instance: JobShopSchedulingProblemInstance
    :param schedule: which attempts to solve the given problem instance. The PotentiallyScheduledOperations
        for each job in the schedule must be ordered in exactly the same way as the corresponding Operations
        in that job
    :type schedule: dict[Job, tuple[PotentiallyScheduledOperation, ...]]
    """

    def __init__(
        self,
        problem_instance: JobShopSchedulingProblemInstance,
        schedule: dict[Job, tuple[PotentiallyScheduledOperation, ...]],
    ):

        if set(problem_instance.jobs) != set(schedule.keys()):
            raise JobShopSchedulingProblemException(
                "The JobShopSchedulingResult must contain the same Jobs "
                + "as the problem instance which it is a solution to!"
            )

        for job in problem_instance.jobs:
            if job.operations != tuple(map(lambda x: x.operation, schedule[job])):
                raise JobShopSchedulingProblemException(
                    "The schedule for a Job must contain the same operations as the Job itself!"
                )

        self._problem_instance: JobShopSchedulingProblemInstance = problem_instance
        self._schedule: dict[Job, tuple[PotentiallyScheduledOperation, ...]] = schedule
        self._is_valid: Optional[bool] = None
        self._makespan: Optional[int] = None

    @property
    def problem_instance(self) -> JobShopSchedulingProblemInstance:
        """
        :return: the problem instance for which this result represents an attempted solution
        :rtype: JobShopSchedulingProblemInstance
        """
        return self._problem_instance

    @property
    def schedule(self) -> dict[Job, tuple[PotentiallyScheduledOperation, ...]]:
        """
        :return: schedule which attempts to solve the given problem instance
        :rtype: dict[Job, tuple[PotentiallyScheduledOperation, ...]]
        """
        return self._schedule

    @property
    def valid_schedule(self) -> dict[Job, tuple[ScheduledOperation, ...]]:
        """
        :return: a schedule which is a valid solution to the given problem instance
        :rtype: dict[Job, tuple[ScheduledOperation, ...]]
        :raises: JobShopSchedulingProblemException if the result itself is not valid
        """
        if self.is_valid:
            # The typeguard in self._is_valid_solution called in self.is_valid ensures that all
            # PotentiallyScheduledOperations are ScheduledOperations
            return cast(dict[Job, tuple[ScheduledOperation, ...]], self._schedule)
        raise JobShopSchedulingProblemException("Cannot access a valid schedule for an invalid result!")

    @property
    def is_valid(self) -> bool:
        """
        :return: true, if the result adheres to all constraints of the JSSP and false otherwise
        :rtype: bool
        """
        if self._is_valid is not None:
            return self._is_valid
        is_valid = self._is_valid_solution()
        self._is_valid = is_valid
        return is_valid

    @property
    def makespan(self) -> Optional[int]:
        """
        :return: the integer makespan if the result is valid and None otherwise
        :rtype: Optional[int]
        """
        if not self.is_valid:
            return None
        if self._makespan is not None:
            return self._makespan
        makespan: int = max(
            (scheduled_operations[-1].end_time for scheduled_operations in self.valid_schedule.values())
        )
        self._makespan = makespan
        return makespan

    def _is_valid_solution(self) -> bool:
        """
        Checks whether this JobShopSchedulingResult is a valid solution. Therefore, it checks
        that all operations are in the correct order and that no operation per job and per machine may
        overlap

        :return: true if the solution is valid and false otherwise
        :rtype: bool
        """

        if not ensure_all_operations_are_scheduled(self._schedule):
            return False

        machine_operation_mapping: dict[Machine, list[ScheduledOperation]] = {
            machine: [] for machine in self._problem_instance.machines
        }

        # Check that all operations per job are correctly ordered and not overlapping
        for job in self._problem_instance.jobs:
            previous_scheduled_operation: Optional[ScheduledOperation] = None
            for scheduled_operation in self._schedule[job]:
                machine_operation_mapping[scheduled_operation.operation.machine].append(scheduled_operation)
                if previous_scheduled_operation is not None:
                    if scheduled_operation.start_time < previous_scheduled_operation.end_time:
                        return False
                previous_scheduled_operation = scheduled_operation

        # Check that all operations per machine are not overlapping
        for scheduled_operations in machine_operation_mapping.values():
            sorted_operations = sorted(scheduled_operations, key=lambda x: x.start_time)
            previous_scheduled_operation = None
            for scheduled_operation in sorted_operations:
                if previous_scheduled_operation is not None:
                    if scheduled_operation.start_time < previous_scheduled_operation.end_time:
                        return False
                previous_scheduled_operation = scheduled_operation

        return True

    def __repr__(self):
        header = f"{self._problem_instance.name} solution with makespan {self.makespan}\n"
        text = ""
        for job in self._problem_instance.jobs:
            text += indent(text=f"{job.name}:\n", prefix=" " * 2)
            for scheduled_operation in self._schedule[job]:
                text += indent(text=f"{str(scheduled_operation)}\n", prefix=" " * 4)
        return header + text


class JobShopSchedulingProblemException(Exception):
    """Exception class representing exceptions caused by invalid job shop scheduling data."""
