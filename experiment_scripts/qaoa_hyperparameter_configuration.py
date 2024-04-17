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
from smac.random_design.probability_design import ProbabilityRandomDesign

from queasars.job_shop_scheduling.serialization import JSSPJSONDecoder
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder


class SPSATerminationChecker:

    def __init__(self, relative_change_threshold, allowed_consecutive_violations):
        self.relative_change_threshold = relative_change_threshold
        self.allowed_consecutive_violations = allowed_consecutive_violations
        self.last_function_value = None
        self.change_history = []
        self.n_function_evaluations = 0
        self.best_eigenvalue = float("inf")
        self.best_parameters = None

    def termination_check(self, n_function_evaluations, parameter_values, function_value, step_size, accepted):

        self.n_function_evaluations = n_function_evaluations

        if self.last_function_value is None:
            self.last_function_value = function_value
            return

        if function_value < self.best_eigenvalue:
            self.best_eigenvalue = function_value
            self.best_parameters = parameter_values

        change = abs(function_value - self.last_function_value) / self.last_function_value
        self.change_history.append(change)
        self.last_function_value = function_value

        if len(self.change_history) < self.allowed_consecutive_violations:
            return False

        if max(self.change_history[-self.allowed_consecutive_violations-1:]) < self.relative_change_threshold:
            return True


def count_controlled_gates(circuit):
    circuit = circuit.decompose()
    controlled_gate_count = 0
    for instr, qargs, _ in circuit.data:
        if instr.name.startswith('c'):
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
            JSSPDomainWallHamiltonianEncoder(jssp_instance=instance[0], makespan_limit=instance[1] + 1),
            instance[1],
        )
        for i, instance in enumerate(problem_instances)
    }
    instance_features = {label: [i] for i, label in enumerate(labeled_instances.keys())}

    def target_function(config: Configuration, instance: str, seed: int):

        QiskitAlgorithmGlobals.random_seed = seed
        sampler_primitive = Sampler(run_options={"seed": seed})

        criterion = SPSATerminationChecker(relative_change_threshold=0.05, allowed_consecutive_violations=5)

        optimizer = SPSA(
            maxiter=2000,
            blocking=bool(config["blocking"]),
            allowed_increase=config["allowed_increase"],
            trust_region=bool(config["trust_region"]),
            perturbation=config["perturbation"],
            learning_rate=config["learning_rate"],
            last_avg=config["last_avg"],
            resamplings=config["resamplings"],
            termination_checker=criterion.termination_check,
        )

        solver = QAOA(sampler=sampler_primitive, optimizer=optimizer)

        hamiltonian = labeled_instances[instance][0].get_problem_hamiltonian()

        result = solver.compute_minimum_eigenvalue(operator=hamiltonian)

        circ = result.optimal_circuit
        circ = circ.assign_parameters(criterion.best_parameters)
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
        Integer("maxiter", (1, 50), default=10, q=1),
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
