# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from functools import reduce
from itertools import permutations

from qiskit.quantum_info.operators import SparsePauliOp

from queasars.job_shop_scheduling.problem_instances import JobShopSchedulingProblemInstance, Machine, Operation


class HamiltonianEncoder:
    jssp_instance: JobShopSchedulingProblemInstance
    time_limit: int

    _encoding_prepared: bool
    _hamiltonian_prepared: bool

    _machine_operations: dict[Machine, list[Operation]]
    _operation_qubit_indices: dict[Operation, tuple[int, int]]
    _operation_start_offset: dict[Operation, int]

    _n_qubits: int
    _local_terms: list[SparsePauliOp]
    _penalty: int

    def __init__(self, jssp_instance: JobShopSchedulingProblemInstance, time_limit: int, penalty: int):
        self.jssp_instance = jssp_instance
        self.time_limit = time_limit
        self._encoding_prepared = False
        self._hamiltonian_prepared = False
        self._machine_operations = {}
        self._operation_qubit_indices = {}
        self._operation_start_offset = {}
        self._n_qubits = 0
        self._local_terms = []
        self._penalty = penalty

    def get_problem_hamiltonian(self) -> SparsePauliOp:
        if not self._encoding_prepared:
            self._prepare_encoding()

        if not self._hamiltonian_prepared:
            self._prepare_hamiltonian()

        return SparsePauliOp.sum(self._local_terms)

    def _prepare_encoding(self) -> None:
        for job in self.jssp_instance.jobs:
            for i, operation in enumerate(job.operations):
                if operation.machine not in self._machine_operations:
                    self._machine_operations[operation.machine] = []
                self._machine_operations[operation.machine].append(operation)

                start_offset = sum(operation.processing_duration for j, operation in enumerate(job.operations) if j < i)
                end_offset = sum(operation.processing_duration for j, operation in enumerate(job.operations) if j >= i)

                self._operation_start_offset[operation] = start_offset
                needed_qubits = self.time_limit - (start_offset + end_offset)

                if needed_qubits > 0:
                    self._operation_qubit_indices[operation] = (self._n_qubits, self._n_qubits + needed_qubits)
                    self._n_qubits += needed_qubits

                if needed_qubits < 0:
                    raise ValueError("There is no feasible solution for the given time_limit!")

        self._encoding_prepared = True

    def _prepare_hamiltonian(self):
        for job in self.jssp_instance.jobs:
            for operation in job.operations:
                viability_local_terms = self._get_domain_wall_viability_local_terms(operation, 10 * self._penalty)
                self._local_terms.extend(viability_local_terms)

            for i in range(0, len(job.operations) - 1):
                precedence_local_terms = self._get_operation_before_local_terms(
                    job.operations[i], job.operations[i + 1], self._penalty
                )
                self._local_terms.extend(precedence_local_terms)

        for operations in self._machine_operations.values():
            if len(operations) < 2:
                continue
            for operation_1, operation_2 in permutations(operations, 2):
                self._local_terms.extend(
                    self._get_operation_overlap_local_terms(operation_1, operation_2, self._penalty)
                )

        self._local_terms.extend(self._get_makespan_optimization_local_terms())

    def _constant_one_local_term(self) -> SparsePauliOp:
        return SparsePauliOp(["I" * self._n_qubits])

    def _pauli_z_local_term(self, qubit_index) -> SparsePauliOp:
        pauli_list = ["I"] * self._n_qubits
        pauli_list[qubit_index] = "Z"
        pauli_string = reduce(lambda x, y: x + y, pauli_list)
        return SparsePauliOp([pauli_string])

    def _domain_wall_z_local_term(
        self, relative_index, domain_wall_qubits_start_index, domain_wall_qubits_end_index
    ) -> SparsePauliOp:
        n_qubits = domain_wall_qubits_end_index - domain_wall_qubits_start_index
        if relative_index < -1 or relative_index > n_qubits:
            raise ValueError("Index is out of the bounds of the domain wall variable!")
        if relative_index == -1:
            return -1 * self._constant_one_local_term()
        if relative_index == n_qubits:
            return self._constant_one_local_term()
        return self._pauli_z_local_term(qubit_index=domain_wall_qubits_start_index + relative_index)

    def _domain_wall_bit_value(
        self, relative_index, domain_wall_qubits_start_index, domain_wall_qubits_end_index
    ) -> SparsePauliOp:
        return (1 / 2) * (
            self._domain_wall_z_local_term(
                relative_index=relative_index,
                domain_wall_qubits_start_index=domain_wall_qubits_start_index,
                domain_wall_qubits_end_index=domain_wall_qubits_end_index,
            )
            - self._domain_wall_z_local_term(
                relative_index=relative_index - 1,
                domain_wall_qubits_start_index=domain_wall_qubits_start_index,
                domain_wall_qubits_end_index=domain_wall_qubits_end_index,
            )
        )

    def _get_domain_wall_viability_local_terms(self, operation: Operation, penalty: int) -> list[SparsePauliOp]:
        if operation not in self._operation_qubit_indices:
            return []

        start_index, end_index = self._operation_qubit_indices[operation]
        n_qubits = end_index - start_index
        penalty = penalty / 2

        local_terms: list[SparsePauliOp] = []
        for i in range(-1, n_qubits):
            local_terms.append(
                1
                / 2
                * penalty
                * (
                    self._constant_one_local_term()
                    - self._domain_wall_z_local_term(
                        relative_index=i,
                        domain_wall_qubits_start_index=start_index,
                        domain_wall_qubits_end_index=end_index,
                    ).compose(
                        self._domain_wall_z_local_term(
                            relative_index=i + 1,
                            domain_wall_qubits_start_index=start_index,
                            domain_wall_qubits_end_index=end_index,
                        )
                    )
                )
            )
        local_terms.append(-1 * penalty * self._constant_one_local_term())

        return local_terms

    def _get_operation_overlap_local_terms(
        self, operation_1: Operation, operation_2: Operation, penalty: int
    ) -> list[SparsePauliOp]:
        offset_1 = self._operation_start_offset[operation_1]
        start_1, end_1 = self._operation_qubit_indices[operation_1]
        n_qubits_1 = end_1 - start_1
        offset_2 = self._operation_start_offset[operation_2]
        start_2, end_2 = self._operation_qubit_indices[operation_2]
        n_qubits_2 = end_2 - start_2

        if offset_1 + n_qubits_1 + operation_1.processing_duration <= offset_2:
            return []

        local_terms = []
        for i_1 in range(0, n_qubits_1 + 1):
            for i_2 in range(0, n_qubits_2 + 1):
                if offset_1 + i_1 <= offset_2 + i_2 < offset_1 + i_1 + operation_1.processing_duration:
                    local_terms.append(
                        penalty
                        * self._domain_wall_bit_value(i_1, start_1, end_1).compose(
                            self._domain_wall_bit_value(i_2, start_2, end_2)
                        )
                    )
        return local_terms

    def _get_operation_before_local_terms(
        self, operation_1: Operation, operation_2: Operation, penalty: int
    ) -> list[SparsePauliOp]:
        offset_1 = self._operation_start_offset[operation_1]
        start_1, end_1 = self._operation_qubit_indices[operation_1]
        n_qubits_1 = end_1 - start_1
        offset_2 = self._operation_start_offset[operation_2]
        start_2, end_2 = self._operation_qubit_indices[operation_2]
        n_qubits_2 = end_2 - start_2

        if offset_1 + n_qubits_1 + operation_1.processing_duration <= offset_2:
            return []

        local_terms = []
        for i_1 in range(0, n_qubits_1 + 1):
            for i_2 in range(0, n_qubits_2 + 1):
                if offset_2 + i_2 < offset_1 + i_1 + operation_1.processing_duration:
                    local_terms.append(
                        penalty
                        * self._domain_wall_bit_value(i_1, start_1, end_1).compose(
                            self._domain_wall_bit_value(i_2, start_2, end_2)
                        )
                    )
        return local_terms

    def _get_makespan_optimization_local_terms(self) -> list[SparsePauliOp]:
        n_jobs = len(self.jssp_instance.jobs)

        max_optimization_value = n_jobs * (n_jobs + 1) ** self.time_limit

        local_terms = []
        for job in self.jssp_instance.jobs:
            last_operation = job.operations[-1]
            start_index, end_index = self._operation_qubit_indices[last_operation]
            n_qubits = end_index - start_index
            start_offset = self._operation_start_offset[last_operation]

            for i in range(0, n_qubits + 1):
                operation_end = start_offset + i + last_operation.processing_duration
                local_terms.append(
                    (1 / max_optimization_value) * (self._penalty - 1)
                    * (n_jobs + 1) ** operation_end
                    * self._domain_wall_bit_value(i, start_index, end_index)
                )
        return local_terms
