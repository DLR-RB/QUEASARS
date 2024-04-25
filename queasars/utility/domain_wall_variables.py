# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from typing import TypeVar, Generic, Hashable, Optional

from qiskit.quantum_info.operators import SparsePauliOp

from queasars.utility.pauli_strings import pauli_identity_string, pauli_z_string


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
        Returns a SparsePauliOp observable, for the ith qubit of this variable.
        For a qubit of this variable (0 <= i < self.n_qubits) this returns the pauli z observable.
        For the virtual qubit before this variable's qubits (i==-1), this returns the negative identity observable.
        For the virtual qubit after this variable's qubits (i==self.n_qubits), this is the positive identity observable.
        For the reasoning behind this see: https://iopscience.iop.org/article/10.1088/2058-9565/ab33c2/meta

        :arg i: relative position within this variable's qubits of the qubit for which to return the _z_dash_term
        :type i: int
        :arg quantum_circuit_n_qubits: the amount of qubits in the quantum circuit in which this variable is part of
        :type quantum_circuit_n_qubits: int
        """
        if i < -1 or i > self.n_qubits:
            raise ValueError("The index is out of the bounds of the domain wall variable!")
        if i == -1:
            return -1 * pauli_identity_string(n_qubits=quantum_circuit_n_qubits)
        if i == self.n_qubits:
            return pauli_identity_string(n_qubits=quantum_circuit_n_qubits)
        return pauli_z_string(qubit_index=self._qubit_start_index + i, n_qubits=quantum_circuit_n_qubits)

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

    def viability_term(self, quantum_circuit_n_qubits: int) -> SparsePauliOp:
        """
        Returns a SparsepauliOp observable which penalizes invalid variable states (states with more than one domain
        wall). Its eigenvalues are 0 only for eigenstates which are valid variable states (contain only one
        domain wall) and (n-1) for eigenstates which contain n domain walls

        :arg quantum_circuit_n_qubits: the amount of qubits in the quantum circuit in which this variable is part of
        :type quantum_circuit_n_qubits: int
        :return: a SparsePauliOp which penalizes invalid variable states
        :rtype: SparsePauliOp
        """
        if self._n_qubits == 0:
            return 0 * pauli_identity_string(n_qubits=quantum_circuit_n_qubits)

        local_terms: list[SparsePauliOp] = []
        for i in range(-1, self._n_qubits):
            local_terms.append(
                1
                / 2
                * (
                    pauli_identity_string(n_qubits=quantum_circuit_n_qubits)
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
        local_terms.append(-1 * pauli_identity_string(n_qubits=quantum_circuit_n_qubits))

        return SparsePauliOp.sum(ops=local_terms)

    def value_term(self, value: T, quantum_circuit_n_qubits: int) -> SparsePauliOp:
        """Returns a SparsePauliOp observable which has an eigenvalue of 0 for all eigenstates in which
        this variable does not choose the given value and 1 for all eigenstates in which this variable
        chooses the given value. If the given value is not within the possible values of this variable,
        this raises a ValueError

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
            return pauli_identity_string(n_qubits=quantum_circuit_n_qubits)

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
        domain_wall_index = self.n_qubits
        for i, value in enumerate(bit_list):
            if value == 0:
                domain_wall_index = i
                break
            if value != 1:
                raise ValueError("The bit_list must only contain 0 or 1 values!")

        if not sum(bit_list[domain_wall_index:]) == 0:
            return None

        return self.values[domain_wall_index]
