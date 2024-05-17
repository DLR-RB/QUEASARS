# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from json import load
from pathlib import Path

from numpy import where

from queasars.job_shop_scheduling.problem_instances import JobShopSchedulingProblemInstance
from queasars.job_shop_scheduling.serialization import JSSPJSONDecoder
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder


def load_benchmarking_dataset() -> dict[int, list[tuple[JobShopSchedulingProblemInstance, int]]]:

    six_operations_file = "six_operations_problem_instances.json"
    with open(Path(Path(__file__).parent, six_operations_file), mode="r") as file:
        six_operation_instances = load(fp=file, cls=JSSPJSONDecoder)

    nine_operations_file = "nine_operations_problem_instances.json"
    with open(Path(Path(__file__).parent, nine_operations_file), mode="r") as file:
        nine_operation_instances = load(fp=file, cls=JSSPJSONDecoder)

    twelve_operations_file = "twelve_operations_problem_instances.json"
    with open(Path(Path(__file__).parent, twelve_operations_file), mode="r") as file:
        twelve_operation_instances = load(fp=file, cls=JSSPJSONDecoder)

    instance_dict = {
        12: six_operation_instances[12],
        15: six_operation_instances[15],
        18: nine_operation_instances[18],
        21: nine_operation_instances[21],
        24: twelve_operation_instances[24],
        28: twelve_operation_instances[28],
    }

    return instance_dict


def get_makespan_energy_split(
    hamiltonian_diagonal,
    encoder: JSSPDomainWallHamiltonianEncoder,
    max_valid_energy: float,
    optimal_makespan: int,
) -> tuple[float, float]:

    indices = where(hamiltonian_diagonal <= max_valid_energy)[0]
    min_boundary = -float("inf")
    max_boundary = float("inf")

    for i in indices:

        energy = hamiltonian_diagonal[i]

        if energy > max_valid_energy or energy > max_boundary or energy < min_boundary:
            continue

        bitstring = format(i, f"0{encoder.n_qubits}b")
        solution = encoder.translate_result_bitstring(bitstring)

        if solution.makespan > optimal_makespan and energy < max_boundary:
            max_boundary = energy

        if solution.makespan == optimal_makespan and energy > min_boundary:
            min_boundary = energy

    return min_boundary, max_boundary
