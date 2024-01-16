# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass
from typing import Optional


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


@dataclass(frozen=True)
class Operation:
    """Dataclass representing an operation in the context of the Job Shop Scheduling Problem. An operation is one
    step in finishing a Job.

    :param name: identifier of the operation. Must not be an empty string
    :type name: str
    :param machine: machine on which this operation must be executed
    :type machine: Machine
    :param processing_duration: amount of time units needed to finish the operation. Must be at least 1
    :type processing_duration: int
    """

    name: str
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

    def __post_init__(self):
        if not self.is_valid():
            raise JobShopSchedulingProblemException("This Operation is invalid and cannot be instantiated!")


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

        if len(set(self.operations)) != len(self.operations):
            return False

        visited_machines: set[Machine] = set()
        for operation in self.operations:

            if operation.machine in visited_machines:
                return False
            visited_machines.add(operation.machine)

            if machines is not None and operation.machine not in machines:
                return False

        return True

    def __post_init__(self):
        if not self.is_valid():
            raise JobShopSchedulingProblemException("This Job is invalid and cannot be instantiated!")


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

        if len(set(self.jobs)) != len(self.jobs):
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


class JobShopSchedulingProblemException(Exception):
    """Exception class representing exceptions caused by invalid job shop scheduling data."""
