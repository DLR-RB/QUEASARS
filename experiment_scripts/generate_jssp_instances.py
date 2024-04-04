# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from collections import Counter
from json import dump
from random import Random
from pathlib import Path

from queasars.job_shop_scheduling.problem_instances import Job, JobShopSchedulingProblemInstance
from queasars.job_shop_scheduling.random_problem_instances import random_job_shop_scheduling_instance
from queasars.job_shop_scheduling.scip_solver import JSSPSCIPModelEncoder
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.job_shop_scheduling.serialization import JSSPJSONEncoder


def get_max_job_length(instance: JobShopSchedulingProblemInstance) -> int:
    return max(sum(operation.processing_duration for operation in job.operations) for job in instance.jobs)


def get_job_hash(j: Job) -> int:
    return hash(tuple(o.processing_duration for o in j.operations))


def get_semantically_unique_hash(instance: JobShopSchedulingProblemInstance) -> int:
    machine_operations = {m: Counter() for m in instance.machines}
    for job in instance.jobs:
        for i, operation in enumerate(job.operations):
            machine_operations[operation.machine].update([(i, operation.processing_duration, get_job_hash(job))])
    hashable = frozenset(frozenset(counter.items()) for counter in machine_operations.values())
    return hash(hashable)


def generate_problem_instances(problem_sizes):
    jssp_problem_sizes = problem_sizes
    max_qubit_size = 30
    relative_op_amount = 1.0
    op_duration = {1: 0.75, 2: 0.25}
    trials_per_problem_size = 1000
    progress_printout = 100

    problems = [[] for _ in range(0, max_qubit_size + 1)]

    for size in jssp_problem_sizes:
        already_seen_instances = set()
        for i in range(0, trials_per_problem_size):
            if i % progress_printout == 0:
                print("problem size: ", problems, ", progress: ", i, "/", trials_per_problem_size)

            random_instance = random_job_shop_scheduling_instance(
                instance_name=f"{size[0]}_jobs_{size[1]}_machines_seed_{i}",
                n_jobs=size[0],
                n_machines=size[1],
                relative_op_amount=relative_op_amount,
                op_duration=op_duration,
                random_seed=i,
            )
            instance_id = get_semantically_unique_hash(instance=random_instance)
            if instance_id in already_seen_instances:
                continue
            already_seen_instances.add(instance_id)

            model_encoder = JSSPSCIPModelEncoder(jssp_instance=random_instance)
            model = model_encoder.get_model()
            model.optimize()
            solution = model.getBestSol()
            result = model_encoder.parse_solution(solution=solution)

            max_job_length = get_max_job_length(instance=random_instance)
            if result.makespan == max_job_length:
                continue

            hamiltonian_encoder = JSSPDomainWallHamiltonianEncoder(
                jssp_instance=random_instance, time_limit=result.makespan + 1
            )
            if hamiltonian_encoder.n_qubits <= max_qubit_size:
                problems[hamiltonian_encoder.n_qubits].append(
                    (random_instance, result.makespan)
                )

    generator = Random(0)
    for i in range(0, max_qubit_size + 1):
        generator.shuffle(problems[i])

    return problems


def main():

    six_op_problems = generate_problem_instances([(2, 3), (3, 2)])

    with open(Path(Path(__file__).parent, "six_operations_problem_instances.json"), mode="w") as file:
        dump(obj=six_op_problems, fp=file, cls=JSSPJSONEncoder, indent=2)

    nine_op_problems = generate_problem_instances([(3, 3)])
    with open(Path(Path(__file__).parent, "nine_operations_problem_instances.json"), mode="w") as file:
        dump(obj=nine_op_problems, fp=file, cls=JSSPJSONEncoder, indent=2)

    twelve_op_problems = generate_problem_instances([(3, 4), (4, 3)])
    with open(Path(Path(__file__).parent, "twelve_operations_problem_instances.json"), mode="w") as file:
        dump(obj=twelve_op_problems, fp=file, cls=JSSPJSONEncoder, indent=2)


if __name__ == "__main__":
    main()
