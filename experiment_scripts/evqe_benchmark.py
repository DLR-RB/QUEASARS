# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from argparse import ArgumentParser
from datetime import datetime
from json import dump
from pathlib import Path
import logging
from sys import stdout
from typing import Optional

from dask.distributed import LocalCluster, Client, wait, warn
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.minimum_eigensolvers.diagonal_estimator import _DiagonalEstimator

from experiment_scripts.benchmark_utility import (
    load_benchmarking_dataset,
    get_makespan_energy_split,
)
from experiment_scripts.benchmark_result_serialization import EVQEBenchmarkResult, ResultEncoder
from queasars.job_shop_scheduling.problem_instances import JobShopSchedulingProblemInstance
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.utility.spsa_termination import SPSATerminationChecker
from queasars.minimum_eigensolvers.evqe.evqe import EVQEMinimumEigensolverConfiguration, EVQEMinimumEigensolver
from queasars.minimum_eigensolvers.base.termination_criteria import BestIndividualRelativeChangeTolerance


def run_single_benchmark(
    benchmark_name: str,
    problem_instance: tuple[JobShopSchedulingProblemInstance, int],
    instance_nr: int,
    seed: int,
    shots: Optional[int],
    qiskit_threads_per_worker: int,
    population_size: int,
) -> bool:

    try:
        problem = problem_instance[0]
        min_makespan = problem_instance[1]

        encoder = JSSPDomainWallHamiltonianEncoder(
            jssp_instance=problem,
            makespan_limit=min_makespan + 1,
            max_opt_value=100,
            encoding_penalty=319,
            overlap_constraint_penalty=319,
            precedence_constraint_penalty=275,
            opt_all_operations_share=0.19,
        )
        hamiltonian = encoder.get_problem_hamiltonian()
        diagonal = hamiltonian.to_matrix(sparse=True).diagonal()

        min_energy = min(diagonal)
        max_energy = max(diagonal)
        max_opt_energy, min_subopt_energy = get_makespan_energy_split(diagonal, encoder, 100, min_makespan)

        if shots is None:
            sampler = Sampler(run_options={"max_parallel_threads": qiskit_threads_per_worker})
        else:
            sampler = Sampler(run_options={"max_parallel_threads": qiskit_threads_per_worker, "shots": shots})
        estimator = _DiagonalEstimator(sampler=sampler, aggregation=0.5)

        checker = SPSATerminationChecker(minimum_relative_change=0.01, allowed_consecutive_violations=2, maxfev=250)
        opt = SPSA(
            maxiter=33,
            blocking=False,
            trust_region=True,
            perturbation=0.35,
            learning_rate=0.43,
            last_avg=1,
            resamplings=1,
            termination_checker=checker.termination_check,
        )
        evqe_termination = BestIndividualRelativeChangeTolerance(
            minimum_relative_change=0.01, allowed_consecutive_violations=1
        )

        with Client(scheduler_file="evqe_scheduler.json") as client:

            logger = logging.getLogger("queasars.minimum_eigensolvers.base.evolving_ansatz_minimum_eigensolver")
            handler = logging.StreamHandler(stream=stdout)
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            config = EVQEMinimumEigensolverConfiguration(
                sampler=sampler,
                estimator=estimator,
                optimizer=opt,
                optimizer_n_circuit_evaluations=66,
                max_generations=None,
                max_circuit_evaluations=15000,
                termination_criterion=evqe_termination,
                random_seed=None,
                population_size=population_size,
                randomize_initial_population_parameters=True,
                n_initial_layers=2,
                speciation_genetic_distance_threshold=1,
                selection_alpha_penalty=0.15,
                selection_beta_penalty=0.02,
                parameter_search_probability=0.39,
                topological_search_probability=0.79,
                layer_removal_probability=0.02,
                parallel_executor=client,
                mutually_exclusive_primitives=False,
                use_tournament_selection=True,
                tournament_size=2,
            )
            evqe = EVQEMinimumEigensolver(configuration=config)

            result = evqe.compute_minimum_eigenvalue(hamiltonian)

            state_translations = {
                state: encoder.translate_result_bitstring(format(state, f"0{hamiltonian.num_qubits}b"))
                for state in result.eigenstate.keys()
            }

            bench_result = EVQEBenchmarkResult(
                n_qubits=hamiltonian.num_qubits,
                instance_nr=instance_nr,
                seed=seed,
                result=result,
                min_penalty=275,
                min_energy=min_energy,
                max_energy=max_energy,
                min_subopt_energy=min_subopt_energy,
                max_opt_energy=max_opt_energy,
                state_translations=state_translations,
                optimal_makespan=min_makespan,
            )

            file_path = Path(
                Path(__file__).parent,
                "benchmarks",
                "evqe",
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
    parser.add_argument("--name", type=str, default=f"qaoa_bench_{datetime.now().isoformat()}", required=False)
    parser.add_argument("--n_workers", type=int, default=10, required=False)
    parser.add_argument("--n_parallel_executions", type=int, default=1, required=False)
    parser.add_argument("--problem_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--instance_indices", type=int, nargs="+", required=True)
    parser.add_argument("--n_runs_per_instance", type=int, default=5, required=False)
    parser.add_argument("--population_size", type=int, default=10, required=False)
    parser.add_argument("--shots", type=int, default=None, required=False)
    parser.add_argument("--memory", type=str, default="2GB", required=False)
    parser.add_argument("--qiskit_threads_per_worker", type=int, default=1, required=False)
    args = parser.parse_args()

    dataset = load_benchmarking_dataset()

    with (
        LocalCluster(
            n_workers=args.n_parallel_executions, threads_per_worker=1, processes=True, memory_limit=args.memory
        ) as scheduling_cluster,
        Client(scheduling_cluster) as scheduling_client,
    ):
        with (
            LocalCluster(
                n_workers=args.n_workers, threads_per_worker=1, processes=True, memory_limit=args.memory
            ) as calculation_cluster,
            Client(calculation_cluster) as calculation_client,
        ):
            calculation_client.write_scheduler_file("evqe_scheduler.json")

            run_confirmations = dict()
            for problem_size in args.problem_sizes:
                for instance_index in args.instance_indices:
                    for seed in range(0, args.n_runs_per_instance):
                        run_confirmations[(problem_size, instance_index, seed)] = scheduling_client.submit(
                            run_single_benchmark,
                            args.name,
                            dataset[problem_size][instance_index],
                            instance_index,
                            seed,
                            args.shots,
                            args.qiskit_threads_per_worker,
                            args.population_size,
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
