import time
from pathlib import Path
from json import load, dump
from argparse import ArgumentParser
from datetime import datetime

from dask.distributed import LocalCluster, Client
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils.algorithm_globals import QiskitAlgorithmGlobals
from qiskit_algorithms.minimum_eigensolvers import QAOA
from smac import Scenario, AlgorithmConfigurationFacade
from smac.multi_objective import ParEGO
from smac.main.config_selector import ConfigSelector

from queasars.job_shop_scheduling.serialization import JSSPJSONDecoder
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.utility.spsa_termination import SPSATerminationChecker


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
    labeled_instances = {
        "instance_"
        + str(i): (
            JSSPDomainWallHamiltonianEncoder(
                jssp_instance=instance[0],
                makespan_limit=instance[1] + 1,
                opt_all_operations_share=0.25,
                constraint_penalty=150,
            ),
            instance[1],
        )
        for i, instance in enumerate(problem_instances)
    }
    instance_features = {label: [i] for i, label in enumerate(labeled_instances.keys())}

    def target_function(config: Configuration, instance: str, seed: int):

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

        solver = QAOA(sampler=sampler_primitive, optimizer=optimizer, aggregation=0.5)

        hamiltonian = labeled_instances[instance][0].get_problem_hamiltonian()

        result = solver.compute_minimum_eigenvalue(operator=hamiltonian)

        circ = result.optimal_circuit
        circ = circ.assign_parameters(criterion.best_parameter_values)
        meas = sampler_primitive.run(circ)
        quasi_distribution = meas.result().quasi_dists[0].binary_probabilities()

        result_value = 0.0
        for bitstring, probability in quasi_distribution.items():
            parsed_result = labeled_instances[instance][0].translate_result_bitstring(bitstring=bitstring)
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

    params = [
        Integer("blocking", (0, 1), default=0, q=1),
        Integer("allowed_increase", (10, 250), q=10),
        Integer("trust_region", (0, 1), default=0, q=1),
        Float("perturbation", (5e-2, 0.5), default=0.1, q=0.05),
        Float("learning_rate", (5e-2, 0.5), default=0.1, q=0.05),
        Integer("last_avg", (1, 4), default=1, q=1),
        Integer("resamplings", (1, 4), default=1, q=1),
    ]
    space = ConfigurationSpace()
    space.add_hyperparameters(params)

    output_directory = Path(Path(__file__).parent, "smac_runs")
    scenario = Scenario(
        space,
        name="qaoa_smac_run_" + args.trial_name,
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

        file_path = Path(Path(__file__).parent, "smac_results", f"qaoa_smac_result_{args.trial_name}.json")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as file:
            dump(obj=serializables, fp=file, indent=2)

    time.sleep(5)


if __name__ == "__main__":
    main()
