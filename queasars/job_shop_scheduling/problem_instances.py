# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass
from typing import Optional
from textwrap import indent


@dataclass(frozen=True)
class Machine:
    """Dataclass representing a machine in the context of the Job Shop Scheduling Problem

    :param name: identifier of the machine
    :type name: str
    """

    name: str

    def is_valid(self) -> bool:
        """Method which checks whether a Machine is valid.
        This specifically checks that the name of the Machine may not be an empty string

        :return: true if the Machine is valid and false otherwise
        :rtype: bool
        """
        return self.name != ""

    def __post_init__(self):
        if not self.is_valid():
            raise JobShopSchedulingProblemException("This Machine is invalid and cannot be instantiated!")

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

    def is_valid(self) -> bool:
        """Method which checks whether an Operation is valid. This specifically checks that the name string
        may not be empty and that the processing duration must be at least 1

        :return: true if the Operation is valid and false otherwise
        :rtype: bool
        """
        if self.name == "":
            return False
        if self.processing_duration <= 0:
            return False
        return True

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
        if not self.is_valid():
            raise JobShopSchedulingProblemException("This Operation is invalid and cannot be instantiated!")

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

    def is_valid(self, machines: Optional[tuple[Machine, ...]] = None) -> bool:
        """Method which checks whether a Job is valid. This specifically checks, that the job name must not be an
        empty string, that the job must consist of at least one operation, that the operations must be unique,
        that no machine is used more than once and if a tuple of usable machines is given,
        that only these machines may be used

        :arg machines: machines which the operations of this job may use. If any operation uses another machine
            not included here, this makes the Job invalid. If no machines are provided, this check is skipped
        :type machines: Optional[tuple[Machine, ...]]

        :return: true if the Job is valid, false otherwise
        :rtype: bool
        """
        if self.name == "":
            return False

        if len(self.operations) == 0:
            return False

        operation_identifiers = set(map(lambda x: x.identifier, self.operations))
        if len(operation_identifiers) != len(self.operations):
            return False

        visited_machines: set[Machine] = set()
        for operation in self.operations:
            if operation.job_name != self.name:
                return False

            if operation.machine in visited_machines:
                return False
            visited_machines.add(operation.machine)

            if machines is not None and operation.machine not in machines:
                return False

        return True

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

    def is_valid(self) -> bool:
        """Method which checks whether a JobShopSchedulingProblemInstance is valid. This specifically checks
        that the problem instance's name may not be an empty string, that the machines anf Jobs must be unique and
        that the jobs only make use of the problem instance's machines

        :return: true if the JobShopProblemInstance is valid, false otherwise
        :rtype: bool
        """
        if self.name == "":
            return False

        if len(set(self.machines)) != len(self.machines):
            return False

        job_names = set(map(lambda x: x.name, self.jobs))
        if len(job_names) != len(self.jobs):
            return False

        for job in self.jobs:
            if not job.is_valid(machines=self.machines):
                return False

        return True

    def __post_init__(self):
        if not self.is_valid():
            raise JobShopSchedulingProblemException(
                "This JobShopProblemInstance is invalid and cannot be instantiated!"
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
class ScheduledOperation:
    """
    Dataclass representing the fact that an Operation has been scheduled to start
    at a certain time

    :param operation: which has been scheduled
    :type operation: Operation
    :param schedule: tuple which contains the time at which the operation is scheduled to start and to end.
        The length of that time interval must match the operation's processing duration!
        A value of None represents the operation not being scheduled or invalidly being scheduled
    :type schedule: Optional[tuple[int, int]]
    """

    operation: Operation
    schedule: Optional[tuple[int, int]]

    def __post_init__(self):
        if self.schedule is not None and self.schedule[1] - self.schedule[0] != self.operation.processing_duration:
            raise JobShopSchedulingProblemException(
                "The schedule of a scheduled operation must match the processing duration of it's operation!"
            )

    def __repr__(self):
        if self.schedule is None:
            return f"{str(self.operation)} was not or invalidly scheduled."
        return f"{str(self.operation)} starts at: {self.schedule[0]} and ends at: {self.schedule[1]}"


@dataclass(frozen=True)
class JobShopSchedulingResult:
    """
    Dataclass representing an attempted solution to the given job shop scheduling problem instance

    :param problem_instance: for which this schedule represents an attempted solution
    :type problem_instance: JobShopSchedulingProblemInstance
    :param schedule: which attempts to solve the given problem instance
    :type schedule: dict[Job, tuple[ScheduledOperation, ...]]
    :param makespan: time after which the last operation of the last job has finished in the schedule
    :type makespan: int
    """

    problem_instance: JobShopSchedulingProblemInstance
    schedule: dict[Job, tuple[ScheduledOperation, ...]]
    makespan: int

    def __repr__(self):
        header = f"{self.problem_instance.name} solution with makespan {self.makespan}\n"
        text = ""
        for job in self.problem_instance.jobs:
            text += indent(text=f"{job.name}:\n", prefix=" " * 2)
            for scheduled_operation in self.schedule[job]:
                text += indent(text=f"{str(scheduled_operation)}\n", prefix=" " * 4)
        return header + text

    def is_valid_solution(self) -> bool:
        """
        Checks whether this JobShopSchedulingResult is a valid solution. Therefore, it checks
        that all operations are in the correct order and that no operation per job and per machine may
        overlap

        :return: true if the solution is valid and false otherwise
        :rtype: bool
        """
        machine_operation_mapping: dict[Machine, list[ScheduledOperation]] = {
            machine: [] for machine in self.problem_instance.machines
        }

        # Check that all operations per job are correctly ordered and not overlapping
        for job in self.problem_instance.jobs:
            previous_scheduled_operation: Optional[ScheduledOperation] = None
            for scheduled_operation in self.schedule[job]:
                machine_operation_mapping[scheduled_operation.operation.machine].append(scheduled_operation)
                # All scheduled operations are checked here before they are accessed.
                # This makes later accesses of schedule typesafe, but mypy cannot verify that.
                if scheduled_operation.schedule is None:
                    return False
                if previous_scheduled_operation is not None:
                    # The schedule can not be none here, ignore mypy!
                    if scheduled_operation.schedule[0] < previous_scheduled_operation.schedule[1]:  # type: ignore
                        return False
                previous_scheduled_operation = scheduled_operation

        # Check that all operations per machine are not overlapping
        for scheduled_operations in machine_operation_mapping.values():
            # The schedule can not be none here, ignore mypy!
            sorted_operations = sorted(scheduled_operations, key=lambda x: x.schedule[0])  # type: ignore
            previous_scheduled_operation = None
            for scheduled_operation in sorted_operations:
                if previous_scheduled_operation is not None:
                    # The schedule can not be none here, ignore mypy!
                    if scheduled_operation.schedule[0] < previous_scheduled_operation.schedule[1]:  # type: ignore
                        return False
                previous_scheduled_operation = scheduled_operation

        return True


class JobShopSchedulingProblemException(Exception):
    """Exception class representing exceptions caused by invalid job shop scheduling data."""
