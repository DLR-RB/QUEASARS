# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from argparse import ArgumentParser
from datetime import datetime
from json import dump
from pathlib import Path

from dask.distributed import LocalCluster, Client, wait, warn
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.minimum_eigensolvers import QAOA

from experiment_scripts.benchmark_utility import (
    load_benchmarking_dataset,
    get_makespan_energy_split,
)
from experiment_scripts.benchmark_result_serialization import VQABenchmarkResult, ResultEncoder
from queasars.job_shop_scheduling.problem_instances import JobShopSchedulingProblemInstance
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.utility.spsa_termination import SPSATerminationChecker


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
            encoding_penalty=307,
            overlap_constraint_penalty=307,
            precedence_constraint_penalty=307,
            opt_all_operations_share=0.33,
        )
        hamiltonian = encoder.get_problem_hamiltonian()
        diagonal = hamiltonian.to_matrix(sparse=True).diagonal()

        min_energy = min(diagonal)
        max_energy = max(diagonal)
        max_opt_energy, _ = get_makespan_energy_split(diagonal, encoder, 100, min_makespan)

        sampler = Sampler()

        checker = SPSATerminationChecker(minimum_relative_change=0.01, allowed_consecutive_violations=10, maxfev=30000)
        opt = SPSA(
            maxiter=15000,
            blocking=True,
            allowed_increase=223,
            trust_region=False,
            perturbation=0.28,
            learning_rate=0.47,
            last_avg=2,
            resamplings=1,
            termination_checker=checker.termination_check,
        )
        qaoa = QAOA(sampler=sampler, optimizer=opt, reps=3, aggregation=0.5)

        result = qaoa.compute_minimum_eigenvalue(hamiltonian)

        circ = result.optimal_circuit
        params = checker.best_parameter_values
        circ = circ.assign_parameters(params)
        sampling_result = sampler.run(circ).result()
        quasi_distribution = sampling_result.quasi_dists[0]

        state_translations = {
            state: encoder.translate_result_bitstring(format(state, f"0{hamiltonian.num_qubits}b"))
            for state, _ in quasi_distribution.keys()
        }

        bench_result = VQABenchmarkResult(
            n_qubits=hamiltonian.num_qubits,
            instance_nr=instance_nr,
            seed=seed,
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
            "qaoa",
            benchmark_name,
            str(hamiltonian.num_qubits),
            str(instance_nr),
            f"result_{seed}.json",
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            dump(obj=bench_result, fp=f, cls=ResultEncoder, indent=2)

    except Exception as e:
        warn(f"The following exception occurred during a benchmarking run:\n {str(e)}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default=f"qaoa_bench_{datetime.now().isoformat()}", required=False)
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
