# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from itertools import combinations
from typing import Optional, Generic, TypeVar, Hashable

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


def _identity_term(n_qubits: int) -> SparsePauliOp:
    """
    Returns a SparsePauliOp observable consisting only of identities. Its expectation value with respect to
    any quantum state is always 1

    :arg n_qubits: number of qubits in the quantum circuit
    :type n_qubits: int
    :return: a SparsePauliOp consisting only of identities
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least one!")
    return SparsePauliOp("I" * n_qubits)


def _pauli_z_term(qubit_index: int, n_qubits: int) -> SparsePauliOp:
    """
    Returns a SparsePauliOp observable consisting of identities and one pauli z at the qubit_index. Its eigenvalues are
    -1 for all eigenstates in which the qubit at the qubit_index is in state |1> and +1 for all eigenstates in which
    the qubit at the qubit_index is in state |0>.

    :arg qubit_index: index of the qubit to which the pauli z term shall apply
    :type qubit_index: int
    :arg n_qubits: number of qubits in the quantum circuit
    :type n_qubits: int
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least one!")

    if not 0 <= qubit_index < n_qubits:
        raise ValueError("The qubit index is invalid!")

    pauli_list: list[str] = ["I"] * n_qubits
    pauli_list[qubit_index] = "Z"
    pauli_string: str = "".join(pauli_list)
    return SparsePauliOp(pauli_string)


T = TypeVar("T", bound=Hashable)


class DomainWallVariable(Generic[T]):
    """
    Class representing a variable encoded in the domain wall encoding. For more details on the domain wall
    encoding see: https://iopscience.iop.org/article/10.1088/2058-9565/ab33c2/meta .
    This class specifically models a choice between n+1 unique values on n qubits. These values must be
    hashable

    :param qubit_start_index: qubit index in the quantum circuit from which this variable starts.
        If values contains n+1 entries, the variable occupies the n qubits in the range
        [qubit_index, ..., qubit_index+n-1]
    :type qubit_start_index: int
    :param values: values between which this variable chooses
    :type values: tuple[T, ...]
    """

    def __init__(self, qubit_start_index: int, values: tuple[T, ...]):
        """Constructor Method"""
        self._qubit_start_index: int = qubit_start_index
        self._values: tuple[T, ...] = values

        if len(self._values) < 1:
            raise ValueError("The domain wall variable must at least have one value!")
        self._value_indices: dict[T, int] = {value: i for i, value in enumerate(self._values)}

        if len(self._values) != len(self._value_indices):
            raise ValueError("All values of a domain wall variable must be unique!")

        self._n_qubits: int = len(values) - 1

    def _z_dash_term(self, i: int, quantum_circuit_n_qubits: int) -> SparsePauliOp:
        """
        Returns a SparsePauliOp, which represents the pauli z operator for the qubits of this variable
        and -1 for the virtual qubit before and 1 for the virtual qubit after the variable's qubits.
        For the reasoning behind this see: https://iopscience.iop.org/article/10.1088/2058-9565/ab33c2/meta
        """
        if i < -1 or i > self.n_qubits:
            raise ValueError("The index is out of the bounds of the domain wall variable!")
        if i == -1:
            return -1 * _identity_term(n_qubits=quantum_circuit_n_qubits)
        if i == self.n_qubits:
            return _identity_term(n_qubits=quantum_circuit_n_qubits)
        return _pauli_z_term(qubit_index=self._qubit_start_index + i, n_qubits=quantum_circuit_n_qubits)

    @property
    def values(self) -> tuple[T, ...]:
        """
        :return: the values between which this domain wall variable chooses
        :rtype: tuple[T, ...]
        """
        return self._values

    @property
    def n_qubits(self) -> int:
        """
        :return: the amount of qubits needed by this variable
        """
        return self._n_qubits

    def viability_term(self, penalty: float, quantum_circuit_n_qubits: int) -> SparsePauliOp:
        """
        Returns a SparsepauliOp which penalizes invalid variable states (states with more than one domain
        wall). Within a hamiltonian this term evaluates to 0 only if the variable is in a valid state and to
        (n-1)*penalty at maximum for the n values this variable can represent

        :arg penalty: size of the applied penalty for each violation
        :type penalty: float
        :arg quantum_circuit_n_qubits: the amount of qubits in the quantum circuit in which this variable is part of
        :type quantum_circuit_n_qubits: int
        :return: a SparsePauliOp which penalizes invalid variable states
        :rtype: SparsePauliOp
        """
        if self._n_qubits == 0:
            return 0 * _identity_term(n_qubits=quantum_circuit_n_qubits)

        penalty = penalty / 2
        local_terms: list[SparsePauliOp] = []
        for i in range(-1, self._n_qubits):
            local_terms.append(
                1
                / 2
                * penalty
                * (
                    _identity_term(n_qubits=quantum_circuit_n_qubits)
                    - self._z_dash_term(
                        i=i,
                        quantum_circuit_n_qubits=quantum_circuit_n_qubits,
                    ).compose(
                        self._z_dash_term(
                            i=i + 1,
                            quantum_circuit_n_qubits=quantum_circuit_n_qubits,
                        )
                    )
                )
            )
        local_terms.append(-1 * penalty * _identity_term(n_qubits=quantum_circuit_n_qubits))

        return SparsePauliOp.sum(ops=local_terms)

    def value_term(self, value: T, quantum_circuit_n_qubits: int) -> SparsePauliOp:
        """Returns a SparsePauliOp which checks the variable for a given value. Within a hamiltonian
        this term evaluates to one only if the variable is in a state which represents the given value and 0
        otherwise. If the given value is not within the possible values of this variable, this raises a ValueError

        :arg value: Value to check the variable for
        :type value: T
        :arg quantum_circuit_n_qubits: amount of qubits in the quantum circuit this variable is part of
        :type quantum_circuit_n_qubits: int
        :return: a SparsePauliOp checking the variable for the given value
        :rtype: SparsePauliOp
        """
        if value not in self._value_indices:
            raise ValueError("The domain wall variable can never assume this value!")
        if self._n_qubits == 0:
            return _identity_term(n_qubits=quantum_circuit_n_qubits)

        i = self._value_indices[value]
        return (1 / 2) * (
            self._z_dash_term(
                i=i,
                quantum_circuit_n_qubits=quantum_circuit_n_qubits,
            )
            - self._z_dash_term(
                i=i - 1,
                quantum_circuit_n_qubits=quantum_circuit_n_qubits,
            )
        )

    def value_from_bitlist(self, bit_list: list[int]) -> Optional[T]:
        """
        Calculates the value held in this domain wall variable for an
        assignment of bit values to the qubits given as a bit_list for the
        whole quantum circuit

        :arg bit_list: representing a value assignment to the quantum circuit's qubits. All list entries must
            be zero or one
        :type bit_list: list[int]
        :return: the value held in this variable
        :rtype: Optional[T]
        """

        bit_list = bit_list[self._qubit_start_index : self._qubit_start_index + self.n_qubits]
        domain_wall_index = len(bit_list)
        for i, value in enumerate(bit_list):
            if value == 0:
                domain_wall_index = i
                break
            if value != 1:
                raise ValueError("The bit_list must only contain 0 or 1 values!")

        if not sum(bit_list[domain_wall_index:]) == 0:
            return None

        return self.values[domain_wall_index]


class JSSPDomainWallHamiltonianEncoder:
    """
    Encoding class used to encode a JobShopSchedulingProblemInstance as a Hamiltonian.
    This uses the Time-indexed model to represent the JSSP with the variables being encoded as domain wall variables

    :param jssp_instance: job shop scheduling problem instance to encode as a hamiltonian
    :type jssp_instance: JobShopSchedulingProblemInstance
    :param time_limit: maximum allowed makespan for possible solutions
    :type time_limit: int
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
        must be in the range (0,1)
    :type opt_all_operations_share: float
    """

    def __init__(
        self,
        jssp_instance: JobShopSchedulingProblemInstance,
        time_limit: int,
        encoding_penalty: float = 300,
        constraint_penalty: float = 100,
        max_opt_value: float = 100,
        opt_all_operations_share: float = 0.25,
    ):
        self.jssp_instance: JobShopSchedulingProblemInstance = jssp_instance
        self.time_limit: int = time_limit
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
        """Counts the needed qubits to encode the problem" and assigns the necessary domain wall variables"""
        for job in self.jssp_instance.jobs:
            for i, operation in enumerate(job.operations):
                if operation.machine not in self._machine_operations:
                    self._machine_operations[operation.machine] = []
                self._machine_operations[operation.machine].append(operation)

                start_offset = sum(operation.processing_duration for j, operation in enumerate(job.operations) if j < i)
                end_offset = sum(operation.processing_duration for j, operation in enumerate(job.operations) if j >= i)

                n_start_times = self.time_limit - (start_offset + end_offset) + 1

                if n_start_times < 1:
                    raise ValueError("There is no feasible solution for the given time_limit!")

                self._operation_start_variables[operation] = DomainWallVariable(
                    qubit_start_index=self._n_qubits,
                    values=tuple(range(start_offset, start_offset + n_start_times)),
                )

                self._n_qubits += self._operation_start_variables[operation].n_qubits

        self._encoding_prepared = True

    def _prepare_hamiltonian(self):
        """
        Gathers the terms making up the hamiltonian. These include penalties for invalid variable states,
        penalties for invalid ordering and overlapping of operations and an optimization term to minimize the
        makespan of the problem
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
                self._local_terms.extend(precedence_term)

        for operations in self._machine_operations.values():
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
        Return a SparsePauliOp which penalizes variable states which make operation_1 and operation_2 overlap.
        Within a hamiltonian this term evaluates to zero if the two operations do not overlap and to penalty if they
        overlap

        :arg operation_1: operation which must not overlap operation_2
        :type operation_1: Operation
        :arg operation_2: operation which must not overlap operation_1
        :type operation_2: Operation
        :arg penalty: penalty value which is applied if the operations overlap
        :type penalty: float
        :return: a SparsePauliOp which penalizes variable states in which the two operations overlap
        :rtype: SparsePauliOp
        """
        start_variable_1 = self._operation_start_variables[operation_1]
        start_variable_2 = self._operation_start_variables[operation_2]

        if max(start_variable_1.values) + operation_1.processing_duration <= min(start_variable_2.values):
            return 0 * _identity_term(n_qubits=self._n_qubits)

        if max(start_variable_2.values) + operation_2.processing_duration <= min(start_variable_1.values):
            return 0 * _identity_term(n_qubits=self._n_qubits)

        local_terms = []
        for start_time_1 in start_variable_1.values:
            for start_time_2 in start_variable_2.values:
                if (
                    start_time_1 <= start_time_2 < start_time_1 + operation_1.processing_duration
                    or start_time_2 <= start_time_1 < start_time_2 + operation_2.processing_duration
                ):
                    local_terms.append(
                        penalty
                        * start_variable_1.value_term(
                            value=start_time_1, quantum_circuit_n_qubits=self._n_qubits
                        ).compose(
                            start_variable_2.value_term(value=start_time_2, quantum_circuit_n_qubits=self._n_qubits)
                        )
                    )
        return SparsePauliOp.sum(local_terms)

    def _operation_precedence_term(
        self, operation_1: Operation, operation_2: Operation, penalty: float
    ) -> SparsePauliOp:
        """
        Returns a SparsePauliOp which penalizes variable states in which operation_2 start before operation_1 ends.
        Within a hamiltonian this term evaluates to zero only if operation_2 starts after (or at the same time) the
        operation_1 has ended. Otherwise, it evaluates to penalty

        :arg operation_1: operation which must precede operation_2
        :type operation_1: Operation
        :arg operation_2: operation which must start after (or at the same time) operation_1 has finished
        :type operation_2: Operation
        :arg penalty: penalty value which is applied if the operation precedence is violated
        :type penalty: float
        :return: a SparsePauliOp which penalizes variable states which violate the operation precedence
        :rtype: SparsePauliOp
        """
        start_variable_1 = self._operation_start_variables[operation_1]
        start_variable_2 = self._operation_start_variables[operation_2]

        if max(start_variable_1.values) + operation_1.processing_duration <= min(start_variable_2.values):
            return 0 * _identity_term(n_qubits=self._n_qubits)

        local_terms = []
        for start_time_1 in start_variable_1.values:
            for start_time_2 in start_variable_2.values:
                if start_time_2 < start_time_1 + operation_1.processing_duration:
                    local_terms.append(
                        penalty
                        * start_variable_1.value_term(
                            value=start_time_1, quantum_circuit_n_qubits=self._n_qubits
                        ).compose(
                            start_variable_2.value_term(value=start_time_2, quantum_circuit_n_qubits=self._n_qubits)
                        )
                    )

        return SparsePauliOp.sum(local_terms)

    def _makespan_optimization_term(self, max_value: float) -> SparsePauliOp:
        """
        Returns a SparsePauliOp which increasingly penalizes the last operation of a job,
        for increasing start times in accordance with the optimization term proposed in
        https://www.sciencedirect.com/science/article/pii/S0377221723002072 .
        Optimizing this term should amount to optimizing the makespan of the JSSP.
        The optimization term is scaled to always be smaller than max_value

        :arg max_value: maximum value of the optimization term
        :type max_value: float
        :return: a SparsePauliOp which penalizes JSSP solutions with higher makespans
        :rtype: SparsePauliOp
        """
        n_jobs = len(self.jssp_instance.jobs)
        max_optimization_value = n_jobs * (n_jobs + 1) ** self.time_limit

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
        Returns a SparsePauliOp which penalizes all operations of all jobs linearly for starting later than
        the earliest possible start time. The optimization term is scaled to always be smaller than max_value

        :arg max_value: maximum value of the optimization term
        :type max_value: float
        :return: a SparsePauliOp which penalizes all operations which start later than necessary
        :rtype: SparsePauliOp
        """
        max_optimization_value = sum(
            (variable.n_qubits**2 + variable.n_qubits) / 2 for variable in self._operation_start_variables.values()
        )

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
