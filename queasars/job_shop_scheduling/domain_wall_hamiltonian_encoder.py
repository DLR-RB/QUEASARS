# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from itertools import combinations

from qiskit.quantum_info.operators import SparsePauliOp

from queasars.job_shop_scheduling.problem_instances import (
    JobShopSchedulingProblemInstance,
    Machine,
    Operation,
    Job,
    PotentiallyScheduledOperation,
    UnscheduledOperation,
    ScheduledOperation,
    JobShopSchedulingResult,
)
from queasars.utility.domain_wall_variables import DomainWallVariable
from queasars.utility.pauli_strings import pauli_identity_string


class JSSPDomainWallHamiltonianEncoder:
    """
    Encoding class used to encode a JobShopSchedulingProblemInstance as a Hamiltonian.
    This uses the Time-indexed model to represent the JSSP with the variables being encoded as domain wall variables

    :param jssp_instance: job shop scheduling problem instance to encode as a hamiltonian
    :type jssp_instance: JobShopSchedulingProblemInstance
    :param makespan_limit: maximum allowed makespan for possible solutions
    :type makespan_limit: int
    :param encoding_penalty: penalty added to the optimization value for violating the encoding constraint
    :type encoding_penalty: float
    :param constraint_penalty: penalty added to the optimization value for violating the JSSP's constraints
    :type constraint_penalty: float
    :param max_opt_value: maximum value of the optimization term. For a clean separation of valid and invalid
        states it should be smaller than both the encoding_penalty and the constraint penalty individually
    :type max_opt_value: float
    :param opt_all_operations_share: the optimization value mostly consists of a term minimizing the makespan.
        This only involves the last operations of each job. If it is desired to also optimize the starting times of
        all other operations, a share of the maximum optimization value can be diverted to that goal. This parameter
        must be in the range [0,1]
    :type opt_all_operations_share: float
    """

    def __init__(
        self,
        jssp_instance: JobShopSchedulingProblemInstance,
        makespan_limit: int,
        encoding_penalty: float = 300,
        constraint_penalty: float = 100,
        max_opt_value: float = 100,
        opt_all_operations_share: float = 0,
    ):
        self.jssp_instance: JobShopSchedulingProblemInstance = jssp_instance
        self.makespan_limit: int = makespan_limit
        self._encoding_prepared: bool = False
        self._hamiltonian_prepared: bool = False
        self._machine_operations: dict[Machine, list[Operation]] = {}
        self._operation_start_variables: dict[Operation, DomainWallVariable[int]] = {}
        self._n_qubits: int = 0
        self._local_terms: list[SparsePauliOp] = []
        self._encoding_penalty: float = encoding_penalty
        self._constraint_penalty: float = constraint_penalty
        self._max_opt_value: float = max_opt_value
        self._opt_all_operations_share: float = opt_all_operations_share

    @property
    def n_qubits(self) -> int:
        """
        :return: the amount of qubits needed to encode this JSSP problem instance
        :rtype: int
        """
        if not self._encoding_prepared:
            self._prepare_encoding()
        return self._n_qubits

    def get_problem_hamiltonian(self) -> SparsePauliOp:
        """
        :return: the problem encoded as a hamiltonian in the form of a SparsePauliOp
        :rtype: SparsePauliOp
        """
        if not self._encoding_prepared:
            self._prepare_encoding()

        if not self._hamiltonian_prepared:
            self._prepare_hamiltonian()

        return SparsePauliOp.sum(self._local_terms)

    def translate_result_bitstring(self, bitstring: str) -> JobShopSchedulingResult:
        """
        Translates a bitstring as measured on a quantum circuit to it's corresponding job shop scheduling result.
        The bitstring must only contain 1 and 0s and must be of length self.n_qubits

        :arg bitstring: string of 1 and 0s measured on the quantum circuit
        :type bitstring: str
        :return: the JobShopSchedulingResult this bitstring represents
        """
        if len(bitstring) != self.n_qubits:
            raise ValueError("The bitstring length does not match the problem size!")

        if not self._encoding_prepared:
            self._prepare_encoding()

        def translate(string: str) -> int:
            if string == "1":
                return 1
            if string == "0":
                return 0
            raise ValueError("The bitstring may not contain any value apart from 1 or 0!")

        bit_list = [translate(char) for char in bitstring]

        job_schedules: dict[Job, tuple[PotentiallyScheduledOperation, ...]] = {}
        for job in self.jssp_instance.jobs:
            scheduled_operations: list[PotentiallyScheduledOperation] = []
            for operation in job.operations:
                domain_wall_variable = self._operation_start_variables[operation]
                start_time = domain_wall_variable.value_from_bitlist(bit_list=bit_list)
                if start_time is not None:
                    scheduled_operations.append(ScheduledOperation(operation=operation, start_time=start_time))
                else:
                    scheduled_operations.append(UnscheduledOperation(operation=operation))
            job_schedules[job] = tuple(scheduled_operations)

        return JobShopSchedulingResult(problem_instance=self.jssp_instance, schedule=job_schedules)

    def _prepare_encoding(self) -> None:
        """Counts the needed qubits to encode the problem and assigns the necessary domain wall variables"""
        for job in self.jssp_instance.jobs:
            # keep track of the start and end bounds for an operation given by the length of the preceding
            # subsequent operations of a job
            start_offset: int = 0
            end_offset: int = sum(operation.processing_duration for operation in job.operations)

            if end_offset > self.makespan_limit:
                raise ValueError(
                    f"There is no feasible solution for the given makespan_limit {self.makespan_limit}!\n"
                    + f"This is due to the length of all operations in job {job.name} which\n"
                    + f"is {end_offset} and is longer than the makespan_limit!"
                )

            for operation in job.operations:
                # cache the operations for each machine
                if operation.machine not in self._machine_operations:
                    self._machine_operations[operation.machine] = []
                self._machine_operations[operation.machine].append(operation)

                # determine all possible start times for the operation and assign a
                # domain wall variable to represent the choice between these start times
                n_start_times = self.makespan_limit - (start_offset + end_offset) + 1
                self._operation_start_variables[operation] = DomainWallVariable(
                    qubit_start_index=self._n_qubits,
                    values=tuple(range(start_offset, start_offset + n_start_times)),
                )

                # keep track of the needed qubits
                self._n_qubits += self._operation_start_variables[operation].n_qubits

                # update the bounds for the next operation
                start_offset += operation.processing_duration
                end_offset -= operation.processing_duration

        self._encoding_prepared = True

    def _prepare_hamiltonian(self):
        """
        Gathers the constituting observables making up the hamiltonian. These include penalties for
        invalid variable states, penalties for invalid ordering and overlapping of operations and an
        optimization term to minimize the makespan of the problem
        """
        for job in self.jssp_instance.jobs:
            for operation in job.operations:
                variable_viability_term = self._operation_start_variables[operation].viability_term(
                    penalty=self._encoding_penalty, quantum_circuit_n_qubits=self._n_qubits
                )
                self._local_terms.append(variable_viability_term)

            for i in range(0, len(job.operations) - 1):
                precedence_term = self._operation_precedence_term(
                    job.operations[i], job.operations[i + 1], self._constraint_penalty
                )
                self._local_terms.append(precedence_term)

        for _, operations in self._machine_operations.items():
            if len(operations) < 2:
                continue
            for operation_1, operation_2 in combinations(operations, 2):
                overlap_term = self._operation_overlap_term(
                    operation_1=operation_1, operation_2=operation_2, penalty=self._constraint_penalty
                )
                self._local_terms.append(overlap_term)

        self._local_terms.append(
            self._makespan_optimization_term(max_value=self._max_opt_value * (1 - self._opt_all_operations_share))
        )
        self._local_terms.append(self._early_start_term(max_value=self._max_opt_value * self._opt_all_operations_share))
        self._hamiltonian_prepared = True

    def _operation_overlap_term(self, operation_1: Operation, operation_2: Operation, penalty: float) -> SparsePauliOp:
        """
        Return a SparsePauliOp observable which penalizes states in which make operation_1 and operation_2 overlap.
        Its eigenvalues are the penalty value for all eigenstates in which operation_1 and operation_2 overlap,
        and 0 for all other eigenstates

        :arg operation_1: operation which should not overlap operation_2
        :type operation_1: Operation
        :arg operation_2: operation which should not overlap operation_1
        :type operation_2: Operation
        :arg penalty: penalty value which is applied if the operations overlap
        :type penalty: float
        :return: a SparsePauliOp which penalizes variable states in which the two operations overlap
        :rtype: SparsePauliOp
        """
        start_variable_1 = self._operation_start_variables[operation_1]
        start_variable_2 = self._operation_start_variables[operation_2]

        # Since the domain wall variables are initialized with their values in ascending order,
        # .values[0] refers to its minimum contained value and .values[-1] refers to its maximum value

        # Check if the operations may never overlap. No penalty is needed then
        if start_variable_1.values[-1] + operation_1.processing_duration <= start_variable_2.values[0]:
            return 0 * pauli_identity_string(n_qubits=self._n_qubits)
        if start_variable_2.values[-1] + operation_2.processing_duration <= start_variable_1.values[0]:
            return 0 * pauli_identity_string(n_qubits=self._n_qubits)

        # Find all start time combinations for which operation_1 and operation_2 overlap
        overlaps: list[tuple[int, int]] = [
            (start_1, start_2)
            for start_1 in start_variable_1.values
            for start_2 in start_variable_2.values
            if start_1 < start_2 + operation_2.processing_duration
            and start_2 < start_1 + operation_1.processing_duration
        ]

        # Penalize the states in which operations overlap
        local_terms = []
        for overlap in overlaps:
            local_terms.append(
                penalty
                * start_variable_1.value_term(value=overlap[0], quantum_circuit_n_qubits=self._n_qubits).compose(
                    start_variable_2.value_term(value=overlap[1], quantum_circuit_n_qubits=self._n_qubits)
                )
            )

        return SparsePauliOp.sum(local_terms)

    def _operation_precedence_term(
        self, operation_1: Operation, operation_2: Operation, penalty: float
    ) -> SparsePauliOp:
        """
        Returns a SparsePauliOp which penalizes states in which operation_2 starts before operation_1 ends.
        Its eigenvalues are the penalty value for all eigenstates in which operation_2 starts before operation_1 has
        finished, and 0 for all other eigenstates

        :arg operation_1: operation which should precede operation_2
        :type operation_1: Operation
        :arg operation_2: operation which should start after (or at the same time) operation_1 has finished
        :type operation_2: Operation
        :arg penalty: penalty value which is applied if the operation precedence is violated
        :type penalty: float
        :return: a SparsePauliOp which penalizes variable states which violate the operation precedence
        :rtype: SparsePauliOp
        """
        start_variable_1 = self._operation_start_variables[operation_1]
        start_variable_2 = self._operation_start_variables[operation_2]

        # Since the domain wall variables are initialized with their values in ascending order,
        # .values[0] refers to its minimum contained value and .values[-1] refers to its maximum value

        # Check if operation 2 may never precede operation 1. No penalty is needed then
        if start_variable_1.values[-1] + operation_1.processing_duration <= start_variable_2.values[0]:
            return 0 * pauli_identity_string(n_qubits=self._n_qubits)

        # Find all start time combinations for which operation 1 does not precede operation 2
        precedence_violations: list[tuple[int, int]] = [
            (start_1, start_2)
            for start_1 in start_variable_1.values
            for start_2 in start_variable_2.values
            if not start_1 + operation_1.processing_duration <= start_2
        ]

        # Penalize the states in which operation 1 does not precede operation 2
        local_terms = []
        for violation in precedence_violations:
            local_terms.append(
                penalty
                * start_variable_1.value_term(value=violation[0], quantum_circuit_n_qubits=self._n_qubits).compose(
                    start_variable_2.value_term(value=violation[1], quantum_circuit_n_qubits=self._n_qubits)
                )
            )

        return SparsePauliOp.sum(local_terms)

    def _makespan_optimization_term(self, max_value: float) -> SparsePauliOp:
        """
        Returns a SparsePauliOp observable which increasingly penalizes the last operation of a job,
        for increasing start times in accordance with the optimization term proposed in
        https://www.sciencedirect.com/science/article/pii/S0377221723002072 .
        Optimizing the expectation value of this observable should amount to optimizing the makespan of the JSSP.
        The eigenvalues of this observable are scaled so that its expectation value is always smaller than max_value

        :arg max_value: maximum value of the optimization term
        :type max_value: float
        :return: a SparsePauliOp which penalizes JSSP solutions with higher makespans
        :rtype: SparsePauliOp
        """
        n_jobs = len(self.jssp_instance.jobs)
        max_optimization_value = n_jobs * (n_jobs + 1) ** self.makespan_limit

        local_terms = []
        for job in self.jssp_instance.jobs:
            last_operation = job.operations[-1]
            start_variable = self._operation_start_variables[last_operation]
            for start_time in start_variable.values:
                operation_end = start_time + last_operation.processing_duration
                local_terms.append(
                    (1 / max_optimization_value)
                    * max_value
                    * (n_jobs + 1) ** operation_end
                    * start_variable.value_term(value=start_time, quantum_circuit_n_qubits=self._n_qubits)
                )

        return SparsePauliOp.sum(local_terms)

    def _early_start_term(self, max_value: float) -> SparsePauliOp:
        """
        Returns a SparsePauliOp observable which penalizes all operations of all jobs linearly for starting later than
        the earliest possible start time. The eigenvalues of this observable are scaled so that its
        expectation value is always smaller than max_value

        :arg max_value: maximum value of the optimization term
        :type max_value: float
        :return: a SparsePauliOp which penalizes all operations which start later than necessary
        :rtype: SparsePauliOp
        """
        max_optimization_value = sum(len(variable.values) - 1 for variable in self._operation_start_variables.values())

        local_terms = []
        for start_variable in self._operation_start_variables.values():
            for i, value in enumerate(start_variable.values):
                if i == 0:
                    continue
                local_terms.append(
                    (1 / max_optimization_value)
                    * max_value
                    * i
                    * start_variable.value_term(value=value, quantum_circuit_n_qubits=self._n_qubits)
                )

        return SparsePauliOp.sum(local_terms)
