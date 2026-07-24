# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from argparse import ArgumentParser
from datetime import datetime
from json import dump
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, wait

from qiskit.primitives import BaseSamplerV1, BackendSampler, PrimitiveJob, SamplerResult
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit_ibm_runtime.fake_provider import FakeOsaka

from experiment_scripts.benchmark_utility import (
    get_makespan_energy_split,
)
from experiment_scripts.benchmark_result_serialization import VQABenchmarkResult, ResultEncoder
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.utility.spsa_termination import SPSATerminationChecker
from queasars.job_shop_scheduling.problem_instances import JobShopSchedulingProblemInstance, Job, Operation, Machine


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


def problem_instance():
    machines = (Machine(name="m0"), Machine(name="m1"))

    j0op1 = Operation(name="j0op0", machine=machines[0], processing_duration=2, job_name="j0")
    j0op2 = Operation(name="j0op1", machine=machines[1], processing_duration=1, job_name="j0")
    job0 = Job(name="j0", operations=(j0op1, j0op2))

    j1op1 = Operation(name="j1op1", machine=machines[0], processing_duration=1, job_name="j1")
    j1op2 = Operation(name="j1op2", machine=machines[1], processing_duration=2, job_name="j1")
    job1 = Job(name="j1", operations=(j1op1, j1op2))

    return JobShopSchedulingProblemInstance(name="Simple Instance", machines=machines, jobs=(job0, job1))


class TranspilingSampler(BaseSamplerV1):
    def __init__(self, sampler: BaseSamplerV1, pass_manager: PassManager):
        super().__init__()
        self._sampler = sampler
        self._pass_manager = pass_manager

    def _run(
        self, circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...], **run_options
    ) -> PrimitiveJob[SamplerResult]:
        applied_circuits = [circuit.assign_parameters(params) for circuit, params in zip(circuits, parameter_values)]
        transpiled_circuits = self._pass_manager.run(applied_circuits)
        return self._sampler.run(transpiled_circuits, **run_options)


def run_single_benchmark(
    benchmark_name: str,
    seed: int,
    shots: int,
    qiskit_threads_per_worker: int,
) -> bool:

    try:
        problem = problem_instance()
        min_makespan = 4

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
        max_opt_energy, min_subopt_energy = get_makespan_energy_split(diagonal, encoder, 100, min_makespan)

        backend = FakeOsaka()
        sampler = BackendSampler(backend=backend, options={"max_parallel_threads": qiskit_threads_per_worker, "shots": shots})
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        sampler = TranspilingSampler(sampler=sampler, pass_manager=pass_manager)

        logging.basicConfig(level=logging.INFO)

        checker = SPSATerminationChecker(
            minimum_relative_change=0.01, allowed_consecutive_violations=9, maxfev=15000, logging_interval=1
        )
        opt = SPSA(
            maxiter=3750,
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
            instance_nr=0,
            seed=seed,
            best_expectation_value=checker.best_function_value,
            best_parameter_values=list(checker.best_parameter_values),
            expectation_evaluation_counts=checker.n_function_evaluation_history,
            expectation_values=checker.function_value_history,
            measurement_distribution=quasi_distribution,
            state_translations=state_translations,
            optimal_makespan=min_makespan,
            min_penalty=806,
            min_energy=min_energy,
            min_subopt_energy=min_subopt_energy,
            max_opt_energy=max_opt_energy,
            max_energy=max_energy,
        )

        file_path = Path(
            Path(__file__).parent,
            "benchmarks",
            "vqe",
            benchmark_name,
            str(hamiltonian.num_qubits),
            str(0),
            f"result_{seed}.json",
        )
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            dump(obj=bench_result, fp=f, cls=ResultEncoder, indent=2)

        return True

    except Exception as e:
        logging.warning(f"The following exception occurred during a benchmarking run:\n {str(e)}")

    return False


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default=f"vqe_bench_{datetime.now().isoformat()}", required=False)
    parser.add_argument("--n_workers", type=int, default=10, required=False)
    parser.add_argument("--qiskit_threads_per_worker", default=2, required=False)
    parser.add_argument("--n_runs", type=int, default=5, required=False)
    parser.add_argument("--shots", type=int, required=True)
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=args.n_workers) as client:
        run_confirmations = dict()
        for seed in range(0, args.n_runs):
            run_confirmations[seed] = client.submit(
                run_single_benchmark,
                args.name,
                seed,
                args.shots,
                args.qiskit_threads_per_worker,
            )

        wait(run_confirmations.values())

        for scenario, result in run_confirmations.items():
            result = result.result()
            if not result:
                logging.warning(
                    f"The benchmark run for the seed {scenario} failed!"
                )


if __name__ == "__main__":
    main()
