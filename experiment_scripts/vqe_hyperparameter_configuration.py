import time
from pathlib import Path
from json import load, dump
from argparse import ArgumentParser
from datetime import datetime

from dask.distributed import LocalCluster, Client
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils.algorithm_globals import QiskitAlgorithmGlobals
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from smac import Scenario, AlgorithmConfigurationFacade
from smac.multi_objective import ParEGO
from smac.main.config_selector import ConfigSelector

from queasars.job_shop_scheduling.serialization import JSSPJSONDecoder
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


def main():

    parser = ArgumentParser()
    parser.add_argument("--trial_name", type=str, default=f"{datetime.now().isoformat()}", required=False)
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument(
        "--n_instances",
        type=int,
        default=5,
        required=False,
    )
    parser.add_argument(
        "--min_budget",
        type=int,
        default=3,
        required=False,
    )
    parser.add_argument(
        "--max_budget",
        type=int,
        default=20,
        required=False,
    )
    parser.add_argument("--n_workers", type=int, default=10, required=False)
    parser.add_argument("--overwrite", type=bool, default=False, required=False)
    args = parser.parse_args()

    problem_instances_file = "six_operations_problem_instances.json"
    with open(Path(Path(__file__).parent, problem_instances_file), mode="r") as file:
        all_problem_instances = load(fp=file, cls=JSSPJSONDecoder)

    problem_instances = all_problem_instances[12][: args.n_instances]
    labeled_instances = {"instance_" + str(i): instance for i, instance in enumerate(problem_instances)}
    instance_features = {label: [i] for i, label in enumerate(labeled_instances.keys())}

    def target_function(config: Configuration, instance: str, seed: int):

        current_instance = labeled_instances[instance]
        encoder = JSSPDomainWallHamiltonianEncoder(
            jssp_instance=current_instance[0],
            makespan_limit=current_instance[1] + 1,
            max_opt_value=100,
            encoding_penalty=config["encoding_penalty"],
            overlap_constraint_penalty=min(config["overlap_constraint_penalty"], config["encoding_penalty"]),
            precedence_constraint_penalty=min(config["precedence_constraint_penalty"], config["encoding_penalty"]),
            opt_all_operations_share=config["opt_all_operations_share"],
        )
        hamiltonian = encoder.get_problem_hamiltonian()

        def solve(hamiltonian):
            QiskitAlgorithmGlobals.random_seed = seed
            sampler_primitive = Sampler(run_options={"seed": seed})

            criterion = SPSATerminationChecker(minimum_relative_change=0.01, allowed_consecutive_violations=4)
            optimizer = SPSA(
                maxiter=1000,
                blocking=bool(config["blocking"]),
                allowed_increase=config["allowed_increase"],
                trust_region=bool(config["trust_region"]),
                perturbation=config["perturbation"],
                learning_rate=config["learning_rate"],
                last_avg=config["last_avg"],
                resamplings=config["resamplings"],
                termination_checker=criterion.termination_check,
            )

            ansatz_circuit = ansatz(n_qubits=hamiltonian.num_qubits, layers=config["repetitions"])
            solver = SamplingVQE(sampler=sampler_primitive, optimizer=optimizer, ansatz=ansatz_circuit, aggregation=0.5)

            result = solver.compute_minimum_eigenvalue(hamiltonian)
            circ = result.optimal_circuit

            try:
                params = criterion.best_parameter_values
            except ValueError:
                return {
                    "result_value": float("inf"),
                    "circuit_evaluations": float("inf"),
                }

            circ = circ.assign_parameters(params)
            meas = sampler_primitive.run(circ)
            quasi_distribution = meas.result().quasi_dists[0].binary_probabilities()

            result_value = 0.0
            for bitstring, probability in quasi_distribution.items():
                parsed_result = encoder.translate_result_bitstring(bitstring=bitstring)
                if not parsed_result.is_valid:
                    result_value += probability * 100
                elif not parsed_result.makespan == labeled_instances[instance][1]:
                    result_value += probability * 50
                else:
                    pass

            return {
                "result_value": result_value,
                "circuit_evaluations": criterion.n_function_evaluations,
            }

        with Client(scheduler_file="vqe_scheduler.json") as client:
            future = client.submit(solve, hamiltonian)
            return future.result()

    params = [
        Integer("blocking", (0, 1), default=0),
        Integer("allowed_increase", (0, 500)),
        Integer("trust_region", (0, 1), default=0),
        Float("perturbation", (1e-2, 0.5), default=0.1),
        Float("learning_rate", (1e-2, 0.5), default=0.1),
        Integer("last_avg", (1, 4), default=1),
        Integer("resamplings", (1, 4), default=1),
        Integer("repetitions", (2, 4), default=2),
        Float("encoding_penalty", (110, 1000), default=300),
        Float("overlap_constraint_penalty", (110, 1000), default=150),
        Float("precedence_constraint_penalty", (110, 1000), default=150),
        Float("opt_all_operations_share", (0, 0.5), default=0.25),
    ]
    space = ConfigurationSpace()
    space.add_hyperparameters(params)

    output_directory = Path(Path(__file__).parent, "smac_runs")
    scenario = Scenario(
        space,
        name="vqe_smac_run_" + args.trial_name,
        objectives=["result_value", "circuit_evaluations"],
        deterministic=False,
        n_trials=args.n_trials,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        instances=list(labeled_instances.keys()),
        instance_features=instance_features,
        output_directory=output_directory,
    )
    selector = ConfigSelector(scenario=scenario, retrain_after=1)

    with (
        LocalCluster(n_workers=args.n_workers, processes=True, threads_per_worker=1) as smac_cluster,
        Client(smac_cluster) as smac_client,
    ):
        with (
            LocalCluster(n_workers=args.n_workers, processes=True, threads_per_worker=1) as calculation_cluster,
            Client(calculation_cluster) as calculation_client,
        ):
            calculation_client.write_scheduler_file("vqe_scheduler.json")

            facade = AlgorithmConfigurationFacade(
                scenario,
                target_function=target_function,
                config_selector=selector,
                multi_objective_algorithm=ParEGO(scenario),
                overwrite=args.overwrite,
                logging_level=10,
                dask_client=smac_client,
            )

            incumbents = facade.optimize()

            if not isinstance(incumbents, list):
                incumbents = [incumbents]

            serializables = []
            for incumbent in incumbents:
                average_cost = facade.runhistory.average_cost(incumbent)
                labeled_cost = list(zip(scenario.objectives, average_cost))
                incumbent_dict = dict(incumbent)
                serializables.append([incumbent.config_id, labeled_cost, incumbent_dict])

            file_path = Path(Path(__file__).parent, "smac_results", f"vqe_smac_result_{args.trial_name}.json")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as file:
                dump(obj=serializables, fp=file, indent=2)

    time.sleep(5)


if __name__ == "__main__":
    main()
