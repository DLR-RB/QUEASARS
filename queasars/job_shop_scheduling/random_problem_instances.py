# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Union, Optional
from random import Random

from queasars.job_shop_scheduling.problem_instances import Machine, Operation, Job, JobShopSchedulingProblemInstance


def _get_random_int_from_range(random_range: Union[int, tuple[int, int]], random_generator: Random) -> int:
    """Returns a random integer given a value range and a random_generator.

    :param random_range: range of values in the form (start, stop) with stop included in the range. It can also
        be a single value, in that case this method without randomness always returns that single value
    :type random_range: Union[int, tuple[int, int]]
    :param random_generator: generator with which to choose the value
    :type random_generator: Random
    :return: a random value from the value_range
    :rtype: int
    """
    if isinstance(random_range, int):
        return random_range
    return random_generator.randrange(start=random_range[0], stop=random_range[1] + 1, step=1)


def _get_random_float_from_range(random_range: Union[float, tuple[float, float]], random_generator: Random) -> float:
    """Returns a random floating point given a value range and a random_generator.

    :param random_range: range of values in the form (start, stop) with stop included in the range. It can also
        be a single value, in that case this method without randomness always returns that single value
    :type random_range: Union[float, tuple[float, float]]
    :param random_generator: generator with which to choose the value
    :type random_generator: Random
    :return: a random value from the value_range
    :rtype: float
    """
    if isinstance(random_range, float):
        return random_range
    return random_generator.random() * (random_range[1] - random_range[0]) + random_range[0]


def random_job_shop_scheduling_instance(
    instance_name: str,
    n_jobs: int,
    n_machines: int,
    relative_op_amount: Union[float, tuple[float, float]],
    op_duration: Union[int, tuple[int, int]],
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
        Can be a value or a value range as (start, stop). It must not exceed the range (0, 1). If a value range
        is given the relative_op_amount is randomly drawn from the value range for each job. The actual
        amount of operations per job is calculated as round(relative_op_amount*n_machines)
    :type relative_op_amount: Union[float, tuple[float, float]
    :arg op_duration: the discrete processing duration assigned to each job. It can be a value or a value range as
        (start, stop). If a value range is given the processing duration is drawn randomly from the value range
        for each Operation
    :type op_duration: Union[int, tuple[int, int]
    :arg random_seed: seed value to control randomness
    :type random_seed: int
    :return: a random job shop scheduling problem instance
    :rtype: JobShopSchedulingProblemInstance
    """
    random_generator: Random = Random(random_seed)

    machines: tuple[Machine, ...] = tuple(Machine(f"m{i}") for i in range(0, n_machines))

    jobs: list[Job] = []
    for i in range(0, n_jobs):
        n_ops = round(
            _get_random_float_from_range(random_range=relative_op_amount, random_generator=random_generator)
            * n_machines
        )
        op_machines = random_generator.sample(population=machines, k=n_ops)
        random_generator.shuffle(op_machines)
        operations = tuple(
            Operation(
                name=f"op{j}",
                job_name=f"job{i}",
                machine=op_machine,
                processing_duration=_get_random_int_from_range(
                    random_range=op_duration, random_generator=random_generator
                ),
            )
            for j, op_machine in enumerate(op_machines)
        )
        jobs.append(Job(name=f"job{i}", operations=operations))

    return JobShopSchedulingProblemInstance(name=instance_name, machines=machines, jobs=tuple(jobs))
