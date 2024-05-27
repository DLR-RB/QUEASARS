# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from qiskit.quantum_info.operators import SparsePauliOp


def pauli_identity_string(n_qubits: int) -> SparsePauliOp:
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


def pauli_z_string(qubit_index: int, n_qubits: int) -> SparsePauliOp:
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
    pauli_list[-(qubit_index + 1)] = "Z"
    pauli_string: str = "".join(pauli_list)
    return SparsePauliOp(pauli_string)
