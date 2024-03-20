# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from queasars.job_shop_scheduling.problem_instances import (
    Machine,
    Operation,
    Job,
    JobShopSchedulingProblemInstance,
    ScheduledOperation,
    JobShopSchedulingResult,
)


def problem_instance() -> JobShopSchedulingProblemInstance:
    m1 = Machine("m1")
    m2 = Machine("m2")

    op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=1)
    op2 = Operation(name="op2", job_name="j1", machine=m2, processing_duration=1)
    j1 = Job(name="j1", operations=(op1, op2))

    op3 = Operation(name="op3", job_name="j2", machine=m2, processing_duration=1)
    op4 = Operation(name="op4", job_name="j2", machine=m1, processing_duration=1)
    j2 = Job(name="j2", operations=(op3, op4))

    return JobShopSchedulingProblemInstance(name="instance", jobs=(j1, j2), machines=(m1, m2))


def valid_result() -> JobShopSchedulingResult:
    instance = problem_instance()

    schedule = {
        instance.jobs[0]: (
            ScheduledOperation(operation=instance.jobs[0].operations[0], start=0),
            ScheduledOperation(operation=instance.jobs[0].operations[1], start=1),
        ),
        instance.jobs[1]: (
            ScheduledOperation(operation=instance.jobs[1].operations[0], start=0),
            ScheduledOperation(operation=instance.jobs[1].operations[1], start=1),
        ),
    }

    return JobShopSchedulingResult(problem_instance=instance, schedule=schedule)


def invalid_result() -> JobShopSchedulingResult:
    instance = problem_instance()

    schedule = {
        instance.jobs[0]: (
            ScheduledOperation(operation=instance.jobs[0].operations[0], start=0),
            ScheduledOperation(operation=instance.jobs[0].operations[1], start=1),
        ),
        instance.jobs[1]: (
            ScheduledOperation(operation=instance.jobs[1].operations[0], start=1),
            ScheduledOperation(operation=instance.jobs[1].operations[1], start=None),
        ),
    }

    return JobShopSchedulingResult(problem_instance=instance, schedule=schedule)
