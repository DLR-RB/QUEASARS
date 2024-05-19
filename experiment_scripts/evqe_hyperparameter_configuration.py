import time
from pathlib import Path
from json import load, dump
from argparse import ArgumentParser
from datetime import datetime
from math import floor

from dask.distributed import LocalCluster, Client
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from qiskit.circuit import QuantumCircuit
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.minimum_eigensolvers.diagonal_estimator import _DiagonalEstimator
from smac import Scenario, AlgorithmConfigurationFacade
from smac.multi_objective import ParEGO
from smac.main.config_selector import ConfigSelector

from queasars.job_shop_scheduling.serialization import JSSPJSONDecoder
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.minimum_eigensolvers.base.termination_criteria import BestIndividualRelativeChangeTolerance
from queasars.minimum_eigensolvers.evqe.evqe import EVQEMinimumEigensolverConfiguration, EVQEMinimumEigensolver
from queasars.utility.spsa_termination import SPSATerminationChecker


def count_controlled_gates(circuit):
    circuit = circuit.decompose()
    controlled_gate_count = 0
    for instr, qargs, _ in circuit.data:
        if instr.name.startswith("c"):
            controlled_gate_count += 1
    return controlled_gate_count


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
    parser.add_argument("--n_parallel_executions", type=int, default=1, required=False)
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

        sampler_primitive = Sampler(run_options={"seed": seed})
        estimator_primitive = _DiagonalEstimator(sampler=sampler_primitive, aggregation=0.5)

        criterion = SPSATerminationChecker(
            minimum_relative_change=0.01,
            allowed_consecutive_violations=config["optimizer_allowed_consecutive_violations"],
            maxfev=250,
        )
        termination_checker = criterion.termination_check

        optimizer = SPSA(
            maxiter=config["maxiter"],
            blocking=bool(config["blocking"]),
            allowed_increase=config["allowed_increase"],
            trust_region=bool(config["trust_region"]),
            perturbation=config["perturbation"],
            learning_rate=config["learning_rate"],
            last_avg=config["last_avg"],
            resamplings=config["resamplings"],
            termination_checker=termination_checker,
        )

        optimizer_n_circuit_evaluations = config["maxiter"] * 2 * config["resamplings"]

        max_generations = None
        max_circuit_evaluations = 30000
        termination_criterion = BestIndividualRelativeChangeTolerance(
            minimum_relative_change=0.01, allowed_consecutive_violations=1
        )

        random_seed = seed

        population_size = 10

        randomize_initial_population_parameters = True

        n_initial_layers = 2

        speciation_genetic_distance_threshold = config["genetic_distance"]

        selection_alpha_penalty = config["alpha_penalty"]
        selection_beta_penalty = config["beta_penalty"]

        parameter_search_probability = config["parameter_search"]

        topological_search_probability = config["topological_search"]

        layer_removal_probability = config["layer_removal"]

        use_tournament_selection = True
        tournament_size = config["tournament_size"]

        with Client(scheduler_file="evqe_scheduler.json") as parallel_executor:
            parallel_executor = parallel_executor

            mutually_exclusive_primitives = False

            configuration = EVQEMinimumEigensolverConfiguration(
                sampler=sampler_primitive,
                estimator=estimator_primitive,
                optimizer=optimizer,
                optimizer_n_circuit_evaluations=optimizer_n_circuit_evaluations,
                max_generations=max_generations,
                max_circuit_evaluations=max_circuit_evaluations,
                termination_criterion=termination_criterion,
                random_seed=random_seed,
                population_size=population_size,
                randomize_initial_population_parameters=randomize_initial_population_parameters,
                n_initial_layers=n_initial_layers,
                speciation_genetic_distance_threshold=speciation_genetic_distance_threshold,
                selection_alpha_penalty=selection_alpha_penalty,
                selection_beta_penalty=selection_beta_penalty,
                parameter_search_probability=parameter_search_probability,
                topological_search_probability=topological_search_probability,
                layer_removal_probability=layer_removal_probability,
                parallel_executor=parallel_executor,
                mutually_exclusive_primitives=mutually_exclusive_primitives,
                use_tournament_selection=use_tournament_selection,
                tournament_size=tournament_size,
            )

            solver = EVQEMinimumEigensolver(configuration=configuration)

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

            result = solver.compute_minimum_eigenvalue(operator=hamiltonian)

            quasi_distribution = result.eigenstate.binary_probabilities()

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
                "circuit_evaluations": result.circuit_evaluations,
                "circuit_depth": result.optimal_circuit.depth(),
                "cx_count": count_controlled_gates(result.optimal_circuit),
            }

    params = [
        Integer("maxiter", (10, 50), default=20),
        Integer("blocking", (0, 1), default=0),
        Float("allowed_increase", (0, 500)),
        Integer("trust_region", (0, 1), default=0),
        Float("perturbation", (1e-2, 0.5), default=0.2),
        Float("learning_rate", (1e-2, 0.5), default=0.2),
        Integer("last_avg", (1, 4), default=1),
        Integer("resamplings", (1, 4), default=1),
        Integer("optimizer_allowed_consecutive_violations", (1, 9), default=2),
        Integer("genetic_distance", (1, 3), default=2),
        Float("alpha_penalty", (0, 0.4), default=0.1),
        Float("beta_penalty", (0, 0.4), default=0.1),
        Float("parameter_search", (0, 0.5), default=0.25),
        Float("topological_search", (0.4, 0.8), default=0.4),
        Float("layer_removal", (0, 0.15), default=0.05),
        Integer("tournament_size", (2, 3), default=2),
        Float("encoding_penalty", (110, 1000), default=300),
        Float("overlap_constraint_penalty", (110, 1000), default=150),
        Float("precedence_constraint_penalty", (110, 1000), default=150),
        Float("opt_all_operations_share", (0.1, 0.5), default=0.5),
    ]
    space = ConfigurationSpace()
    space.add_hyperparameters(params)

    output_directory = Path(Path(__file__).parent, "smac_runs")
    scenario = Scenario(
        space,
        name="evqe_smac_run_" + args.trial_name,
        objectives=["result_value", "circuit_evaluations", "circuit_depth", "cx_count"],
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
        LocalCluster(n_workers=args.n_parallel_executions, processes=True, threads_per_worker=1) as smac_cluster,
        Client(smac_cluster) as smac_client,
    ):
        with (
            LocalCluster(n_workers=args.n_workers, processes=True, threads_per_worker=1) as calculation_cluster,
            Client(calculation_cluster) as calculation_client,
        ):
            calculation_client.write_scheduler_file("evqe_scheduler.json")

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

            file_path = Path(Path(__file__).parent, "smac_results", f"evqe_smac_result_{args.trial_name}.json")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as file:
                dump(obj=serializables, fp=file, indent=2)

    time.sleep(5)


if __name__ == "__main__":
    main()