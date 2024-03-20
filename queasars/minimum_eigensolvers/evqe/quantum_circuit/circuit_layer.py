# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass
from random import Random
from typing import Optional

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.converters import circuit_to_gate

from queasars.minimum_eigensolvers.evqe.quantum_circuit.quantum_gate import (
    EVQEGate,
    EVQEGateType,
    IdentityGate,
    RotationGate,
    ControlGate,
    ControlledGate,
    ControlledRotationGate,
)


@dataclass(frozen=True)
class EVQECircuitLayer:
    """Dataclass representing a single circuit layer in quantum circuit.
    This class is intended for structuring the genome and not for implementing quantum circuit functionality

    :param n_qubits: The number of qubits on which this circuit layer is applied
    :type n_qubits: int
    :param gates: The ordered gates contained in this circuit layer, with the tuple indices matching the qubit indices.
        The tuples length must match n_qubits
    :type gates: tuple[EVQEGate, ...]
    """

    n_qubits: int
    gates: tuple[EVQEGate, ...]

    @staticmethod
    def random_layer(
        n_qubits: int,
        previous_layer: Optional["EVQECircuitLayer"] = None,
        random_seed: Optional[int] = None,
    ) -> "EVQECircuitLayer":
        """
        Creates a random circuit layer. If a previous layer is given, the layer generation is adapted to
        prevent adding unnecessary parameters.

        :arg n_qubits: amount of qubits to which this layer shall be applied
        :type n_qubits: int
        :arg previous_layer: optional previous layer to restrict layer generation by
        :type previous_layer: EVQECircuitLayer
        :arg random_seed: integer value to control randomness
        :type random_seed: Optional[int]
        :return: the randomly generated circuit layer
        :rtype: EVQECircuitLayer
        """

        if n_qubits < 1:
            raise EVQECircuitLayerException("A circuit layer may not have fewer than one qubit!")

        # Ensure, that the previous layer matches the layer to be generated
        if previous_layer is not None and previous_layer.n_qubits != n_qubits:
            raise EVQECircuitLayerException(
                f"The previous_layer has {previous_layer.n_qubits} qubits which differs from the {n_qubits} "
                + "for the layer which shall be randomly generated! The amount of qubits for both layers must match!"
            )

        # Initialize a buffer to hold the gates and parameters for the new layer
        chosen_gates: list[EVQEGate] = list(IdentityGate(qubit_index=qubit_index) for qubit_index in range(0, n_qubits))
        controlled_rotation_qubits: list[int] = []

        random_generator = Random(random_seed)

        # Iterate over each qubit and randomly either assign a rotation
        # or mark the qubit for use in a controlled rotation
        for qubit_index in range(0, n_qubits):
            # If the previous layer held a rotation or identity gate here,
            # this layer may only hold a controlled rotation here
            if previous_layer is not None and previous_layer.gates[qubit_index].gate_type() in [
                EVQEGateType.ROTATION,
                EVQEGateType.IDENTITY,
            ]:
                # Controlled rotations are placed at the end, so this is only noted
                controlled_rotation_qubits.append(qubit_index)

            # If there is no previous layer or the previous layer held a controlled rotation,
            # both a rotation or a controlled rotation can be placed
            else:
                gate_type = random_generator.choice([EVQEGateType.ROTATION, EVQEGateType.CONTROLLED_ROTATION])
                if gate_type == EVQEGateType.CONTROLLED_ROTATION:
                    # Controlled rotations are placed at the end, so this is only noted
                    controlled_rotation_qubits.append(qubit_index)
                elif gate_type == EVQEGateType.ROTATION:
                    # Rotations can already be placed
                    chosen_gates[qubit_index] = RotationGate(qubit_index=qubit_index)

        # Since each controlled rotation consists of a ControlGate and a ControlledGate there need to be
        # at least two free qubits to place it
        while len(controlled_rotation_qubits) >= 2:
            # Choose a random qubit as control and controlled qubit respectively
            rotation_qubit, control_qubit = random_generator.sample(controlled_rotation_qubits, 2)
            rotation_gate = ControlledRotationGate(qubit_index=rotation_qubit, control_qubit_index=control_qubit)
            control_gate = ControlGate(qubit_index=control_qubit, controlled_qubit_index=rotation_qubit)

            # Check that there is no matching controlled rotation in the previous layer
            # to prevent adding unnecessary parameters.
            if (
                previous_layer is not None
                and rotation_gate not in previous_layer.gates
                and control_gate not in previous_layer.gates
            ) or previous_layer is None:
                chosen_gates[control_qubit] = control_gate
                chosen_gates[rotation_qubit] = rotation_gate
                controlled_rotation_qubits.remove(rotation_qubit)
                controlled_rotation_qubits.remove(control_qubit)

        # On the last remaining qubit no controlled rotation can be placed.
        # Instead, place a rotation if possible otherwise leave it as an identity gate
        if len(controlled_rotation_qubits) == 1:
            qubit_index = controlled_rotation_qubits[0]
            if previous_layer is not None and previous_layer.gates[qubit_index].gate_type() == EVQEGateType.ROTATION:
                chosen_gates[qubit_index] = IdentityGate(qubit_index=qubit_index)
            else:
                chosen_gates[qubit_index] = RotationGate(qubit_index=qubit_index)

        return EVQECircuitLayer(n_qubits=n_qubits, gates=tuple(chosen_gates))

    def __post_init__(self) -> None:
        # Buffer the amount of parameters offered by this circuit layer to prevent recalculating it
        object.__setattr__(self, "_n_parameters", int(sum(gate.n_parameters() for gate in self.gates)))

        # Buffer the amount of controlled gates to prevent recounting them
        n_controlled_gates = sum(1 for gate in self.gates if isinstance(gate, ControlledGate))
        object.__setattr__(self, "_n_controlled_gates", n_controlled_gates)

        # Disallow the creation of invalid circuit layers
        if not self.is_valid():
            raise EVQECircuitLayerException("The created layer is invalid!")

    @property
    def n_parameters(self) -> int:
        """
        :return: The number of parameters offered by this circuit layer
        :rtype: int
        """
        # This attribute is set in __post_init__ which mypy and pylint do not recognize.
        return self._n_parameters  # type: ignore # pylint: disable=no-member

    @property
    def n_controlled_gates(self) -> int:
        """
        :return: The number of controlled gates in this circuit layer
        :rtype: int
        """
        # This attribute is set in __post_init__ which mypy and pylint do not recognize.
        return self._n_controlled_gates  # type: ignore # pylint: disable=no-member

    def is_valid(self) -> bool:
        """Checks whether this circuit layer is valid

        :return: True, if it is valid, False otherwise
        :rtype: bool
        """

        # The circuit layer must have exactly as many gates as qubits
        if len(self.gates) != self.n_qubits:
            return False

        # Check the validity of each gate
        for gate_index, gate in enumerate(self.gates):
            # Ensure each gate is at the right position in the gates tuple
            if gate_index != gate.qubit_index:
                return False

            # Ensure each ControlledGate has a correctly assigned ControlGate
            if isinstance(gate, ControlledGate):
                control_gate = self.gates[gate.control_qubit_index]
                if not (isinstance(control_gate, ControlGate) and control_gate.controlled_qubit_index == gate_index):
                    return False

            # Ensure each ControlGate has a correctly assigned ControlledGate
            if isinstance(gate, ControlGate):
                controlled_gate = self.gates[gate.controlled_qubit_index]
                if not (
                    isinstance(controlled_gate, ControlledRotationGate)
                    and controlled_gate.control_qubit_index == gate_index
                ):
                    return False

        return True

    def get_parameterized_layer_circuit(self, layer_id: int) -> QuantumCircuit:
        """Creates a parameterized quantum circuit which contains only this circuit layer

        :arg layer_id: id used in parameter and gate naming, should be unique for all layers of a quantum circuit
        :type layer_id: int
        :return: a parameterized quantum circuit containing only this circuit layer
        :rtype: QuantumCircuit
        """
        circuit = QuantumCircuit(self.n_qubits, name=f"layer_{layer_id}")

        layer_prefix = f"layer{layer_id}_"
        for gate in self.gates:
            gate.apply_gate(circuit=circuit, parameter_name_prefix=layer_prefix)

        return circuit

    def get_parameterized_layer_gate(self, layer_id: int) -> Gate:
        """Creates a parameterized qiskit Gate which can apply this circuit layer to any quantum circuit

        :arg layer_id: id used in parameter and gate naming, should be unique for all layers of a quantum circuit
        :type layer_id: int
        :return: a qiskit Gate operator with which this layer can be applied
        :rtype: Gate
        """
        return circuit_to_gate(self.get_parameterized_layer_circuit(layer_id=layer_id))

    def get_layer_gate(self, layer_id: int, parameter_values: tuple[float, ...]) -> Gate:
        """Creates a qiskit Gate which can apply this circuit layer to any quantum circuit using the given parameters

        :arg layer_id: id used in parameter and gate naming, should be unique for all layers of a quantum circuit
        :type layer_id: int
        :arg parameter_values: parameter values to be applied in the circuit layer
        :type parameter_values: tuple[float, ...]
        :return: a qiskit Gate operator with which this layer can be applied
        :rtype: Gate
        """
        # Make sure the correct amount of parameter values is provided
        if len(parameter_values) != self.n_parameters:
            raise EVQECircuitLayerException(
                "The amount of provided parameter values must match this layer's n_parameters!"
            )

        circuit = self.get_parameterized_layer_circuit(layer_id=layer_id)
        circuit.assign_parameters(parameters=parameter_values, inplace=True)
        return circuit_to_gate(circuit)


class EVQECircuitLayerException(Exception):
    """Class for exceptions caused during operations involving the EVQECircuitLayer"""
