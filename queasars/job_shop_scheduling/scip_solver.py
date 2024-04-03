# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import Optional
from itertools import combinations

from pyscipopt import Model
from pyscipopt.scip import Variable, Solution

from queasars.job_shop_scheduling.problem_instances import (
    Machine,
    Job,
    Operation,
    JobShopSchedulingProblemInstance,
    JobShopSchedulingResult,
    PotentiallyScheduledOperation,
    ScheduledOperation,
)


class JSSPSCIPModelEncoder:
    """
    Encoding class used to encode a JobShopSchedulingProblemInstance as a pyscipopt SCIP solver model.
    Needs queasars to be installed with the pyscipopt extra: pip install queasars[pyscipopt].
    It also needs SCIP to be installed in the version 8.*. In some cases, this
    should happen automatically with the pip install, others might need a manual installation.
    For details see https://github.com/scipopt/PySCIPOpt/blob/master/INSTALL.md .
    A manual installation of scip can for instance be done with conda:
    ``conda install conda-forge::scip``

    :param jssp_instance: job shop scheduling problem instance to encode as a scip model
    :type jssp_instance: JobShopSchedulingProblemInstance
    """

    def __init__(self, jssp_instance: JobShopSchedulingProblemInstance):

        self._jssp_instance: JobShopSchedulingProblemInstance = jssp_instance

        self._machine_operations: dict[Machine, list[Operation]] = {
            machine: [] for machine in self._jssp_instance.machines
        }
        for job in self._jssp_instance.jobs:
            for operation in job.operations:
                self._machine_operations[operation.machine].append(operation)

        self._model: Model = Model()
        self._variables_prepared: bool = False
        self._optimization_var: Optional[Variable] = None
        self._operation_start_variables: dict[Operation, Variable] = {}
        self._constraints_prepared: bool = False

    def get_model(self) -> Model:
        """
        Sets up a pyscipopt SCIP optimization model. Subsequent calls return the cached model

        :return: a pyscipopt SCIP optimization model
        :rtype: Model
        """
        if not self._variables_prepared:
            self._prepare_variables()

        if not self._constraints_prepared:
            self._prepare_constraints()

        return self._model

    def parse_solution(self, solution: Solution) -> JobShopSchedulingResult:
        """
        Parses a pyscipopt solution for this encoder's optimization model

        :param solution: solution obtained by the SCIP solver
        :type solution: Solution
        :return: a JobShopSchedulingResult parsed from the solution
        :rtype: JobShopSchedulingResult
        """
        job_schedules: dict[Job, tuple[PotentiallyScheduledOperation, ...]] = {}
        for job in self._jssp_instance.jobs:
            scheduled_operations: list[PotentiallyScheduledOperation] = []
            for operation in job.operations:
                start_time: int = int(solution[self._operation_start_variables[operation]])
                scheduled_operations.append(ScheduledOperation(operation=operation, start_time=start_time))
            job_schedules[job] = tuple(scheduled_operations)

        return JobShopSchedulingResult(problem_instance=self._jssp_instance, schedule=job_schedules)

    def _prepare_variables(self):
        """
        Creates the integer variables for the starting times of each operation and constrains them to be at least 0.
        Also creates an integer variable to hold the makespan, which is constrained to be larger than the end times
        of the last operations. Sets the optimization goal to minimize the makespan variable. Does nothing
        if the variables have already been prepared
        """

        if self._variables_prepared:
            return

        # create auxiliary variable to hold the makespan as an optimization goal
        self._optimization_var = self._model.addVar("optimization_var", vtype="INTEGER", lb=0)
        self._model.setObjective(self._optimization_var)

        for job in self._jssp_instance.jobs:
            for operation in job.operations:
                # Create an integer variable for the start time of each operation
                # Constrain it to be at least 0
                var = self._model.addVar(operation.identifier, vtype="INTEGER", lb=0)
                self._operation_start_variables[operation] = var

        self._variables_prepared = True

    def _prepare_constraints(self):
        """
        Constrains all operations within each job to start in order and not overlap.
        Constrains all operations for each machine to not overlap. To this end auxiliary binary variables
        are created to enable ordering decisions. Does nothing, if the constraints have already been prepared
        """

        if self._constraints_prepared:
            return

        if not self._variables_prepared:
            self._prepare_variables()

        for job in self._jssp_instance.jobs:
            n_operations = len(job.operations)

            # Ensure by constraint, that each operation in a job finishes, before the next one may start
            for i in range(0, n_operations - 1):
                self._model.addCons(
                    self._operation_start_variables[job.operations[i]] + job.operations[i].processing_duration
                    <= self._operation_start_variables[job.operations[i + 1]],
                    name=f"Ensure operation {job.operations[i].identifier} precedes {job.operations[i + 1].identifier}"
                    + f" in job {job.name}.",
                )
            # For each job the makespan must be larger than the end time of its last operation
            self._model.addCons(
                self._operation_start_variables[job.operations[n_operations - 1]]
                + job.operations[n_operations - 1].processing_duration
                <= self._optimization_var,
                name=f"Ensure that the makespan is larger than {job.operations[n_operations - 1].identifier} end time"
                + f" in job {job.name}.",
            )

        # Ensure for each machine, that no operation may overlap
        for machine, operations in self._machine_operations.items():

            # For a machine look at all possible combination of two operations
            for operation_1, operation_2 in combinations(operations, 2):
                # Introduce a binary variable to allow the choice of precedence between the two operations
                order_var = self._model.addVar(
                    f"order_{operation_1.identifier}_{operation_2.identifier}", vtype="BINARY"
                )
                # Depending on the precedence order, ensure that the later operation may only start, once the earlier
                # one has finished
                self._model.addCons(
                    order_var * (self._operation_start_variables[operation_1] + operation_1.processing_duration)
                    <= self._operation_start_variables[operation_2],
                    name=f"Ensure operation {operation_1.identifier} precedes "
                    + f"{operation_2.identifier} on machine {machine.name}.",
                )
                self._model.addCons(
                    (1 - order_var) * (self._operation_start_variables[operation_2] + operation_2.processing_duration)
                    <= self._operation_start_variables[operation_1],
                    name=f"Ensure operation {operation_2.identifier} precedes "
                    + f"{operation_1.identifier} on machine {machine.name}.",
                )

        self._constraints_prepared = True
