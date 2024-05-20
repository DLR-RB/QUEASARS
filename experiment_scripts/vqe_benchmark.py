# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from argparse import ArgumentParser
from datetime import datetime
from json import dump
from pathlib import Path
from sys import stdout
import logging

from dask.distributed import LocalCluster, Client, wait, warn
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE

from experiment_scripts.benchmark_utility import (
    load_benchmarking_dataset,
    get_makespan_energy_split,
)
from experiment_scripts.benchmark_result_serialization import VQABenchmarkResult, ResultEncoder
from queasars.job_shop_scheduling.problem_instances import JobShopSchedulingProblemInstance
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.utility.spsa_termination import SPSATerminationChecker


def ansatz(n_qubits, layers):
    circuit = QuantumCircuit(n_qubits)
    name_counter = 0

    for i in range(0, n_qubits):
        param = Parameter(name=f"\u03B8{name_counter}")
        name_counter += 1
        circuit.ry(param, i)

    circuit.barrier()

    for i in range(0, layers):

        for j in range(0, n_qubits - 1):
            if j % 2 == 0:
                circuit.cx(control_qubit=j, target_qubit=(j + 1))

        for j in range(0, n_qubits - 1):
            if (j + 1) % 2 == 0:
                circuit.cx(control_qubit=j, target_qubit=(j + 1))

        circuit.barrier()

        for j in range(0, n_qubits):
            param = Parameter(name=f"\u03B8{name_counter}")
            name_counter += 1
            circuit.ry(param, j)

        circuit.barrier()

    return circuit


def run_single_benchmark(
    benchmark_name: str, problem_instance: tuple[JobShopSchedulingProblemInstance, int], instance_nr: int, seed: int
) -> bool:

    try:
        problem = problem_instance[0]
        min_makespan = problem_instance[1]

        encoder = JSSPDomainWallHamiltonianEncoder(
            jssp_instance=problem,
            makespan_limit=min_makespan + 1,
            max_opt_value=100,
            encoding_penalty=841,
            overlap_constraint_penalty=806,
            precedence_constraint_penalty=841,
            opt_all_operations_share=0.15,
        )
        hamiltonian = encoder.get_problem_hamiltonian()
        diagonal = hamiltonian.to_matrix(sparse=True).diagonal()

        min_energy = min(diagonal)
        max_energy = max(diagonal)
        max_opt_energy, _ = get_makespan_energy_split(diagonal, encoder, 100, min_makespan)

        sampler = Sampler()

        logger = logging.getLogger("queasars.utility.spsa_termination")
        handler = logging.StreamHandler(stdout)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        checker = SPSATerminationChecker(
            minimum_relative_change=0.01, allowed_consecutive_violations=9, maxfev=30000, logging_interval=50
        )
        opt = SPSA(
            maxiter=7500,
            blocking=False,
            trust_region=True,
            perturbation=0.21,
            learning_rate=0.26,
            last_avg=4,
            resamplings=2,
            termination_checker=checker.termination_check,
        )
        ansatz_circuit = ansatz(n_qubits=hamiltonian.num_qubits, layers=2)
        vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_circuit, optimizer=opt, aggregation=0.5)

        result = vqe.compute_minimum_eigenvalue(hamiltonian)

        circ = result.optimal_circuit
        params = checker.best_parameter_values
        circ = circ.assign_parameters(params)
        sampling_result = sampler.run(circ).result()
        quasi_distribution = sampling_result.quasi_dists[0]

        state_translations = {
            state: encoder.translate_result_bitstring(format(state, f"0{hamiltonian.num_qubits}b"))
            for state in quasi_distribution.keys()
        }

        bench_result = VQABenchmarkResult(
            n_qubits=hamiltonian.num_qubits,
            instance_nr=instance_nr,
            seed=seed,
            best_expectation_value=checker.best_function_value,
            best_parameter_values=list(checker.best_parameter_values),
            expectation_evaluation_counts=checker.n_function_evaluation_history,
            expectation_values=checker.function_value_history,
            measurement_distribution=quasi_distribution,
            state_translations=state_translations,
            optimal_makespan=min_makespan,
            min_energy=min_energy,
            max_opt_energy=max_opt_energy,
            max_energy=max_energy,
        )

        file_path = Path(
            Path(__file__).parent,
            "benchmarks",
            "vqe",
            benchmark_name,
            str(hamiltonian.num_qubits),
            str(instance_nr),
            f"result_{seed}.json",
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            dump(obj=bench_result, fp=f, cls=ResultEncoder, indent=2)

        return True

    except Exception as e:
        warn(f"The following exception occurred during a benchmarking run:\n {str(e)}")

    return False


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default=f"vqe_bench_{datetime.now().isoformat()}", required=False)
    parser.add_argument("--n_workers", type=int, default=10, required=False)
    parser.add_argument("--problem_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--instance_indices", type=int, nargs="+", required=True)
    parser.add_argument("--n_runs_per_instance", type=int, default=5, required=False)
    parser.add_argument("--memory", type=str, default="2GB", required=False)
    args = parser.parse_args()

    dataset = load_benchmarking_dataset()

    with (
        LocalCluster(
            n_workers=args.n_workers, threads_per_worker=1, processes=True, memory_limit=args.memory
        ) as CalculationCluster,
        Client(CalculationCluster) as client,
    ):

        run_confirmations = dict()
        for problem_size in args.problem_sizes:
            for instance_index in args.instance_indices:
                for seed in range(0, args.n_runs_per_instance):
                    run_confirmations[(problem_size, instance_index, seed)] = client.submit(
                        run_single_benchmark,
                        args.name,
                        dataset[problem_size][instance_index],
                        instance_index,
                        seed,
                    )

        wait(run_confirmations.values())

        for scenario, result in run_confirmations.items():
            result = result.result()
            if not result:
                warn(
                    f"The benchmark run for the instance of size {scenario[0]} with \n"
                    + f"the index {scenario[1]} and seed {scenario[2]} failed!"
                )


if __name__ == "__main__":
    main()
