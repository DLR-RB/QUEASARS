# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from math import isclose
from random import Random
from typing import Union, Optional, TypeVar

from queasars.job_shop_scheduling.problem_instances import Machine, Operation, Job, JobShopSchedulingProblemInstance


T = TypeVar("T")


def _get_random_value_from_distribution(distribution: dict[T, float], random_generator: Random) -> T:
    """
    Returns a random element from a probability distribution with the according likelihood.

    :param distribution: dictionary in which the keys are the elements to be drawn from and the values are the
        probabilities. The probabilities of all entries must add up to one
    :type distribution: dict[T, float]
    :param random_generator: random generator which seeds the randomness
    :type random_generator: Random
    :return: a random element from the distribution
    :rtype: T
    """
    if not isclose(sum(distribution.values()), 1, abs_tol=0.001):
        raise ValueError("The probabilities in the distribution should add up to 1!")
    return random_generator.choices(population=list(distribution.keys()), weights=list(distribution.values()), k=1)[0]


def _get_value(value_or_distribution: Union[T, dict[T, float]], random_generator: Random) -> T:
    """
    Returns a random element of the distribution, ff the given value is a probability distribution,
    and the value itself, if it is not.

    :param value_or_distribution: probability distribution to draw from or basic value to return
    :type value_or_distribution: Union[T, dict[T, float]
    :param random_generator: random generator which seeds the randomness
    :type random_generator: Random
    :return:
    """
    if isinstance(value_or_distribution, dict):
        return _get_random_value_from_distribution(
            distribution=value_or_distribution, random_generator=random_generator
        )
    return value_or_distribution


def random_job_shop_scheduling_instance(
    instance_name: str,
    n_jobs: int,
    n_machines: int,
    relative_op_amount: Union[float, dict[float, float]],
    op_duration: Union[int, dict[int, float]],
    random_seed: Optional[int] = None,
):
    """
    Generates a random job shop scheduling problem instance of the name instance_name, with n_jobs and n_machines.
    Does NOT guarantee that a Machine must actually be assigned a job

    :arg instance_name: name of the problem instance to be generated
    :type instance_name: str
    :arg n_jobs: amount of jobs to be scheduled
    :type n_jobs: int
    :arg n_machines: amount of machines on which job operations might occur. This is the upper bound of the amount
        of machines which are actually used
    :type n_machines: int
    :arg relative_op_amount: relative amount of operations per job in relation to the amount of machines.
        Can be a value or a probability distribution of values as a dictionary, in which the keys are the operations per
        job and the values the probability with which this amount of operations per job occurs.
    :type relative_op_amount: Union[float, dict[float, float]]
    :arg op_duration: the discrete processing duration assigned to each job. It can be a value or a probability
        distribution as dictionary, in which the keys are the processing duration and the values the probability
        with which this processing duration occurs
    :type op_duration: Union[int, dict[int, float]]
    :arg random_seed: optional seed value to control randomness
    :type random_seed: Optional[int]
    :return: a random job shop scheduling problem instance
    :rtype: JobShopSchedulingProblemInstance
    """
    random_generator: Random = Random(random_seed)

    machines: tuple[Machine, ...] = tuple(Machine(f"m{i}") for i in range(0, n_machines))

    jobs: list[Job] = []
    for i in range(0, n_jobs):
        n_ops = round(_get_value(relative_op_amount, random_generator) * n_machines)
        op_machines = random_generator.sample(population=machines, k=n_ops)
        random_generator.shuffle(op_machines)
        operations = tuple(
            Operation(
                name=f"op{j}",
                job_name=f"job{i}",
                machine=op_machine,
                processing_duration=_get_value(value_or_distribution=op_duration, random_generator=random_generator),
            )
            for j, op_machine in enumerate(op_machines)
        )
        jobs.append(Job(name=f"job{i}", operations=operations))

    return JobShopSchedulingProblemInstance(name=instance_name, machines=machines, jobs=tuple(jobs))
