# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import CU3Gate


class EVQEGateType(Enum):
    """Enum representing the type of gate which can be placed for each qubit.
    This class is intended for structuring the genome and not for implementing quantum gate functionality
    """

    IDENTITY = 0
    ROTATION = 1
    CONTROL = 2
    CONTROLLED_ROTATION = 3


@dataclass(frozen=True)
class EVQEGate(ABC):
    """Abstract dataclass representing a quantum gate.
    This class is intended for structuring the genome and not for implementing quantum gate functionality

    :param qubit_index: Index of the qubit on which this gate is placed
    :type qubit_index: int
    """

    qubit_index: int

    @staticmethod
    @abstractmethod
    def gate_type() -> EVQEGateType:
        """
        :return: the EVQEGateType of this gate
        :rtype: EVQEGateType
        """

    @staticmethod
    @abstractmethod
    def n_parameters() -> int:
        """
        :return: the amount of parameters this gate offers
        :rtype: int
        """

    @abstractmethod
    def apply_gate(self, circuit: QuantumCircuit, parameter_name_prefix: str) -> None:
        """
        Applies this gate inplace to the given QuantumCircuit. If this gate offers
        parameters, it is applied parameterized, with the parameter_name_prefix
        being prepended to each parameter's name

        :arg circuit: Circuit to which the gate is applied inplace.
        :type circuit: QuantumCircuit
        :arg parameter_name_prefix: to prepend to each parameter name
        :type parameter_name_prefix: str
        """


@dataclass(frozen=True)
class IdentityGate(EVQEGate):
    """Dataclass representing an identity gate.
    This class is intended for structuring the genome and not for implementing quantum gate functionality
    """

    @staticmethod
    def gate_type() -> EVQEGateType:
        return EVQEGateType.IDENTITY

    @staticmethod
    def n_parameters() -> int:
        return 0

    def apply_gate(self, circuit: QuantumCircuit, parameter_name_prefix: str) -> None:
        circuit.id(self.qubit_index)


@dataclass(frozen=True)
class RotationGate(EVQEGate):
    """Dataclass representing a rotation gate.
    This class is intended for structuring the genome and not for implementing quantum gate functionality
    """

    @staticmethod
    def gate_type() -> EVQEGateType:
        return EVQEGateType.ROTATION

    @staticmethod
    def n_parameters() -> int:
        return 3

    def apply_gate(self, circuit: QuantumCircuit, parameter_name_prefix: str) -> None:
        circuit.u(
            theta=Parameter(parameter_name_prefix + f"q{self.qubit_index}_theta"),
            phi=Parameter(parameter_name_prefix + f"q{self.qubit_index}_phi"),
            lam=Parameter(parameter_name_prefix + f"q{self.qubit_index}_lambda"),
            qubit=self.qubit_index,
        )


@dataclass(frozen=True)
class ControlGate(EVQEGate):
    """Dataclass representing the controlling part of a controlled quantum gate. It is not a valid gate on its own.
    It needs a matching ControlledGate to be placed at the qubit of the controlled_qubit_index.
    This class is intended for structuring the genome and not for implementing quantum gate functionality

    :param controlled_qubit_index: The index of the qubit on which the belonging ControlledGate is placed
    :type controlled_qubit_index: int
    """

    controlled_qubit_index: int

    @staticmethod
    def gate_type() -> EVQEGateType:
        return EVQEGateType.CONTROL

    @staticmethod
    def n_parameters() -> int:
        return 0

    def apply_gate(self, circuit: QuantumCircuit, parameter_name_prefix: str) -> None:
        pass


@dataclass(frozen=True)
class ControlledGate(EVQEGate, ABC):
    """Dataclass representing a controlled quantum gate. It is not a valid gate on its own.
    It needs a matching ControlGate to be placed at the qubit of the control_qubit_index.
    This class is intended for structuring the genome and not for implementing quantum gate functionality

    :param control_qubit_index: The index of the qubit on which the belonging ControlGate is placed
    :type control_qubit_index: int
    """

    control_qubit_index: int


@dataclass(frozen=True)
class ControlledRotationGate(ControlledGate):
    """Dataclass representing a ControlledGate which applies a rotation to the qubit.
    It needs a matching ControlGate to be placed at the qubit of the control_qubit_index.
    This class is intended for structuring the genome and not for implementing quantum gate functionality
    """

    @staticmethod
    def gate_type() -> EVQEGateType:
        return EVQEGateType.CONTROLLED_ROTATION

    @staticmethod
    def n_parameters() -> int:
        return 3

    def apply_gate(self, circuit: QuantumCircuit, parameter_name_prefix: str) -> None:
        circuit.append(
            instruction=CU3Gate(
                theta=Parameter(parameter_name_prefix + f"q{self.qubit_index}_theta"),
                phi=Parameter(parameter_name_prefix + f"q{self.qubit_index}_phi"),
                lam=Parameter(parameter_name_prefix + f"q{self.qubit_index}_lambda"),
            ),
            qargs=(self.control_qubit_index, self.qubit_index),
        )
