from pathlib import Path
import os
from json import load

from dask.distributed import LocalCluster, Client
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer
from qiskit_aer.primitives import Sampler, Estimator
from qiskit_algorithms.optimizers import SPSA
from smac import Scenario, AlgorithmConfigurationFacade
from smac.multi_objective import ParEGO

from queasars.job_shop_scheduling.serialization import JSSPJSONDecoder
from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.minimum_eigensolvers.base.termination_criteria import PopulationChangeRelativeTolerance
from queasars.minimum_eigensolvers.evqe.evqe import EVQEMinimumEigensolverConfiguration, EVQEMinimumEigensolver


def main():
    problem_instances_file = "six_operations_problem_instances.json"
    with open(Path(Path(__file__).parent, problem_instances_file), mode="r") as file:
        all_problem_instances = load(fp=file, cls=JSSPJSONDecoder)

    def target_function(config: Configuration, instance: str, seed: int):

        # The EVQEMinimumEigensolver needs a sampler and can also use an estimator.
        # Here we use the sampler and estimator provided by the qiskit_aer simulator.
        sampler_primitive = Sampler()
        estimator_primitive = Estimator()

        # The EVQEMinimumEigensolver also needs a qiskit optimizer. It should be
        # configured to terminate quickly, so that mutations are not overtly expensive.
        # Here we use the SPSA optimizer with a very limited amount of iterations and a
        # large step size.
        optimizer = SPSA(
            maxiter=config["maxiter"],
            blocking=bool(config["blocking"]),
            trust_region=bool(config["trust_region"]),
            perturbation=config["perturbation"],
            learning_rate=config["learning_rate"],
            last_avg=config["last_avg"],
            resamplings=config["resamplings"],
        )

        # To help the EVQEMinimumEigensolver deal correctly with terminations based
        # on the amount of circuit evaluations used, an estimate can be given for how
        # many circuit evaluations the optimizer uses per optimization run.
        # SPSA makes two measurements per sampling, which means in total it will
        # need 48 circuit evaluations for 12 iterations with 2 resamplings.
        optimizer_n_circuit_evaluations = config["maxiter"] * 2 * config["resamplings"]

        # To specify when the EVQEMinimumEigensolver should terminate either max_generations,
        # max_circuit_evaluations or a termination_criterion should be given.
        # Here we choose to terminate once the best individual changes by less than 5%
        # in expectation value per generation.
        max_generations = 20
        max_circuit_evaluations = None
        termination_criterion = PopulationChangeRelativeTolerance(
            minimum_relative_change=0.05, allowed_consecutive_violations=2
        )

        # A random seed can be provided to control the randomness of the evolutionary process.
        random_seed = None

        # The population size determines how many individuals are evaluated each generation.
        # With a higher population size, fewer generations might be needed, but this also
        # makes each generation more expensive to evaluate. A reasonable range might be
        # 10 - 100 individuals per population. Here we use a population size of 10.
        population_size = 10

        # If the optimization algorithm can't deal with parameter values of 0 at the beginning
        # of the optimization, they can be randomized here. For this example we don't need this.
        randomize_initial_population_parameters = True

        # Determines how many circuit layers apart two individuals need to be, to be considered to
        # be of a different species. Reasonable values might be in the range 2 - 5. Here we use 3.
        speciation_genetic_distance_threshold = config["genetic_distance"]

        # The alpha and beta penalties penalize quantum circuits of increasing depth (alpha) and
        # increasing amount of controlled rotations (beta). increase them if the quantum circuits get to
        # deep or complicated. For now we will use values of 0.1 for both penalties.
        selection_alpha_penalty = 2
        selection_beta_penalty = 0.2

        # The parameter search probability determines how likely an individual is mutated by optimizing
        # all it's parameter values. This should not be too large as this is costly. Here we will use
        # a probability of 0.24.
        parameter_search_probability = config["parameter_search"]

        # The topological search probability determines how likely a circuit layer is added to an individual
        # as a mutation. Here we will use a probability of 0.2
        topological_search_probability = config["topological_search"]

        # The layer removal probability determines how likely circuit layers are removed from an individual
        # as a mutation. This is a very disruptive mutation and should only be used sparingly to counteract
        # circuit growth. Here we will use a probability of 0.05
        layer_removal_probability = config["layer_removal"]

        # An executor for launching parallel computation can be specified.
        # This can be a dask Client or a python ThreadPoolExecutor. If None is
        # specified a ThreadPoolExecutor with population_size many threads will
        # be used
        parallel_executor = Client(scheduler_file="scheduler.json")

        # Discerns whether to only allow mutually exclusive access to the Sampler and
        # Estimator primitive respectively. This is needed if the Sampler or Estimator are not threadsafe and
        # a ThreadPoolExecutor with more than one thread or a Dask Client with more than one thread per process is used.
        # For safety reasons this is enabled by default. If the sampler and estimator are threadsafe disabling this
        # option may lead to performance improvements
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
            speciation_genetic_distance_threshold=speciation_genetic_distance_threshold,
            selection_alpha_penalty=selection_alpha_penalty,
            selection_beta_penalty=selection_beta_penalty,
            parameter_search_probability=parameter_search_probability,
            topological_search_probability=topological_search_probability,
            layer_removal_probability=layer_removal_probability,
            parallel_executor=parallel_executor,
            mutually_exclusive_primitives=mutually_exclusive_primitives,
        )

        solver = EVQEMinimumEigensolver(configuration=configuration)

        hamiltonian = labeled_instances[instance][0].get_problem_hamiltonian()
        result = solver.compute_minimum_eigenvalue(operator=hamiltonian)

        quasi_distribution = result.eigenstate.binary_probabilities()

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
            "circuit_evaluations": result.circuit_evaluations,
            "circuit_depth": result.optimal_circuit.depth(),
        }

    problem_instances = all_problem_instances[12][:5]
    labeled_instances = {
        "instance_"
        + str(i): (
            JSSPDomainWallHamiltonianEncoder(jssp_instance=instance[0], makespan_limit=instance[1] + 1),
            instance[1],
        )
        for i, instance in enumerate(problem_instances)
    }
    instance_features = {"instance_" + str(i): [i] for i in range(0, 5)}

    params = [
        Integer("maxiter", (1, 50), default=10, q=1),
        Integer("blocking", (0, 1), default=0, q=1),
        Integer("trust_region", (0, 1), default=0, q=1),
        Float("perturbation", (5e-2, 0.5), default=0.1, q=0.05),
        Float("learning_rate", (5e-2, 0.5), default=0.1, q=0.05),
        Integer("last_avg", (1, 4), default=1, q=1),
        Integer("resamplings", (1, 4), default=1, q=1),
        Integer("genetic_distance", (1, 5), default=2, q=1),
        Float("parameter_search", (0, 1), default=0.25, q=0.05),
        Float("topological_search", (0, 1), default=0.4, q=0.05),
        Float("layer_removal", (0, 1), default=0.05, q=0.05),
    ]
    space = ConfigurationSpace()
    space.add_hyperparameters(params)

    scenario = Scenario(
        space,
        objectives=["result_value", "circuit_evaluations", "circuit_depth"],
        deterministic=False,
        n_trials=10,
        min_budget=2,
        max_budget=10,
        instances=list(labeled_instances.keys()),
        instance_features=instance_features,
    )

    with LocalCluster(n_workers=2, processes=True, threads_per_worker=1) as smac_cluster:
        with smac_cluster.get_client() as smac_client:
            with LocalCluster(n_workers=20, processes=True, threads_per_worker=1) as calculation_cluster:
                with calculation_cluster.get_client() as calculation_client:
                    calculation_client.write_scheduler_file("scheduler.json")

                    facade = AlgorithmConfigurationFacade(
                        scenario,
                        target_function=target_function,
                        multi_objective_algorithm=ParEGO(scenario),
                        overwrite=True,
                        logging_level=10,
                        dask_client=smac_client,
                    )

                    incumbent = facade.optimize()


if __name__ == "__main__":
    main()
