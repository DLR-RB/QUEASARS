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
    ControlledRotationGate, RZGate, ControlledZGate, EchoedCrossResonanceGate, SXGate, XGate,
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
            all_possible_gates_weighted: dict[EVQEGateType | tuple[EVQEGateType, EVQEGateType], float],
            coupling_map: Optional[list[tuple[int, int]]],
            previous_layer: Optional["EVQECircuitLayer"] = None,
            random_seed: Optional[int] = None,
    ) -> "EVQECircuitLayer":
        """
        New implementation with more gates.

        Creates a random circuit layer. If a previous layer is given, the layer generation is adapted to
        prevent adding unnecessary parameters.

        :arg n_qubits: amount of qubits to which this layer shall be applied
        :type n_qubits: int
        :arg all_possible_gates_weighted: the allowed (single qubit) gate types or two-element tuples of gate type combinations,
         with the respective weight factor for the random sampling
        :arg previous_layer: optional previous layer to restrict layer generation by
        :type previous_layer: EVQECircuitLayer
        :arg random_seed: integer value to control randomness
        :type random_seed: Optional[int]
        :return: the randomly generated circuit layer
        :rtype: EVQECircuitLayer
        """
        random_generator = Random(random_seed)

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
        qubits_todo: set[int] = set(range(0, n_qubits))
        all_possible_gates: list[EVQEGateType | tuple[EVQEGateType, EVQEGateType]] = list(
            all_possible_gates_weighted.keys())
        incompatible_gate_combinations: dict[EVQEGateType, list[EVQEGateType]] = {
            EVQEGateType.ROTATION: [EVQEGateType.ROTATION, EVQEGateType.RZ],
            EVQEGateType.RZ: [EVQEGateType.ROTATION, EVQEGateType.RZ],
        }

        def get_gate_for_type(gate_type: EVQEGateType, qubit_index: int, partner_index: Optional[int]) -> EVQEGate:
            if gate_type is EVQEGateType.IDENTITY:
                return IdentityGate(qubit_index)
            elif gate_type is EVQEGateType.ROTATION:
                return RotationGate(qubit_index)
            elif gate_type is EVQEGateType.SX:
                return SXGate(qubit_index)
            elif gate_type is EVQEGateType.X:
                return XGate(qubit_index)
            elif gate_type is EVQEGateType.RZ:
                return RZGate(qubit_index)
            else:
                # Gates with a partner
                if partner_index is None:
                    raise ValueError(f"partner_index must not be None for {gate_type=}")
                if gate_type is EVQEGateType.CONTROL:
                    return ControlGate(qubit_index, partner_index)
                elif gate_type is EVQEGateType.CONTROLLED_ROTATION:
                    return ControlledRotationGate(qubit_index, partner_index)
                elif gate_type is EVQEGateType.CZ:
                    return ControlledZGate(qubit_index, partner_index)
                elif gate_type is EVQEGateType.ECR:
                    return EchoedCrossResonanceGate(qubit_index, partner_index)
            raise ValueError(f"Unknown gate type: {gate_type}")

        def get_coupleable_qubits_todo(index: int, coupling_map: Optional[list[tuple[int, int]]], qubits_todo: set[int]) -> \
                set[int]:
            """
            Returns all qubits from qubits_todo that can be directly coupled with a specific qubit.
             If the coupling_map is None, all qubits from qubits_todo are returned.
            """
            if coupling_map is None:
                return qubits_todo
            coupleable_indices: set[int] = set()
            for (control, controlled) in coupling_map:
                if control == index and controlled in qubits_todo:
                    coupleable_indices.add(controlled)
                elif controlled == index and control in qubits_todo:
                    coupleable_indices.add(control)
            return coupleable_indices

        def get_incompatible_gates(qubit_index: int) -> set[EVQEGateType]:
            """
            Returns all gates that are invalid for a specific qubit,
            due to the gate that was used for that qubit in the previous layer.
            """
            incompatible_gates = set()
            if previous_layer is not None:
                previous_gate = previous_layer.gates[qubit_index].gate_type()
                if previous_gate in incompatible_gate_combinations.keys():
                    incompatible_gates = set(incompatible_gate_combinations[previous_gate])
            return incompatible_gates

        def get_possible_gates_for_qubit(qubit_index: int, qubits_todo: set[int], is_primary_qubit: bool) -> set[
            EVQEGateType | tuple[EVQEGateType, EVQEGateType]]:
            """
            Returns all possible gates for a specific qubit.
            When a previous layer exists, gates that would be invalid in combination with the previous layer are never returned.
            When no remaining qubits are left, only valid single-qubit gates are returned.
            """
            possible_gates = all_possible_gates.copy()
            incompatible_gates = set()
            if previous_layer is not None:
                incompatible_gates: set[EVQEGateType] = get_incompatible_gates(qubit_index)
                possible_gates = {gate for gate in possible_gates if
                                  (isinstance(gate, EVQEGateType) and gate not in incompatible_gates or (
                                          isinstance(gate, tuple) and (
                                          is_primary_qubit and gate[0] not in incompatible_gates) or
                                          not is_primary_qubit and gate[1] not in incompatible_gates
                                  )
                                   )}
            if len(qubits_todo) == 0:
                # We have no remaining qubit to couple with, use only single-qubit gates
                single_qubit_gates = {gate for gate in possible_gates if not isinstance(gate, tuple)}
                compatible_single_qubit_gates: set[EVQEGateType] = single_qubit_gates.difference(incompatible_gates)
                return compatible_single_qubit_gates
            return possible_gates

        while len(qubits_todo) >= 1:
            current_qubit: int = random_generator.sample(list(qubits_todo), k=1)[0]
            qubits_todo.remove(current_qubit)

            chosen_gate: Optional[EVQEGate] = None
            possible_gates_this_qubit: set[
                EVQEGateType | tuple[EVQEGateType, EVQEGateType]] = get_possible_gates_for_qubit(current_qubit,
                                                                                                 qubits_todo,
                                                                                                 is_primary_qubit=True)
            while chosen_gate is None and len(possible_gates_this_qubit) > 0:
                # Try all possible gates until we find either a single-qubit gate, or a two-qubit gate and a valid partner.

                possible_gates_this_qubit_list = list(possible_gates_this_qubit)
                # TODO prefer two-qubit gates?
                chosen_gate_candidate: EVQEGateType | tuple[EVQEGateType, EVQEGateType] = \
                    random_generator.choices(possible_gates_this_qubit_list, k=1,
                                             weights=[all_possible_gates_weighted[gate] for gate in
                                                      possible_gates_this_qubit_list])[0]

                if isinstance(chosen_gate_candidate, tuple):
                    # If a two-qubit gate was chosen, try to find a partner
                    self_gate_type = chosen_gate_candidate[0]
                    partner_gate_type = chosen_gate_candidate[1]
                    possible_partner_qubits = []
                    # TODO encourage all qubits to couple with at least one other based on previous layer (or better all previous layers)
                    for possible_partner_qubit in get_coupleable_qubits_todo(current_qubit, coupling_map, qubits_todo):
                        if partner_gate_type not in get_incompatible_gates(possible_partner_qubit):
                            possible_partner_qubits.append(possible_partner_qubit)
                    if len(possible_partner_qubits) == 0:
                        # We cannot find a partner, therefore this gate combination is not possible anymore
                        possible_gates_this_qubit.remove(chosen_gate_candidate)
                    else:
                        partner_qubit_index: int = random_generator.sample(possible_partner_qubits, k=1)[0]
                        chosen_gate = get_gate_for_type(self_gate_type, qubit_index=current_qubit,
                                                        partner_index=partner_qubit_index)
                        partner_gate = get_gate_for_type(partner_gate_type, qubit_index=partner_qubit_index,
                                                         partner_index=current_qubit)
                        chosen_gates[partner_qubit_index] = partner_gate
                        qubits_todo.remove(partner_qubit_index)
                else:
                    chosen_gate = get_gate_for_type(chosen_gate_candidate, qubit_index=current_qubit,
                                                    partner_index=None)
            if chosen_gate is not None:
                # If we found a gate for this qubit, set it; if no possible gates are left for this qubit, we skip it
                chosen_gates[current_qubit] = chosen_gate

        # TODO validate here (as we know the valid gates here)?
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
                gate_types = (gate.gate_type(), control_gate.gate_type())
                # if gate_types not in self.all_possible_gates and (gate_types[1], gate_types[0]) not in self.all_possible_gates:
                #     return False
                if isinstance(control_gate, ControlGate) and not control_gate.controlled_qubit_index == gate_index:
                    return False

            # Ensure each ControlGate has a correctly assigned ControlledGate
            if isinstance(gate, ControlGate):
                controlled_gate = self.gates[gate.controlled_qubit_index]
                gate_types = (gate.gate_type(), controlled_gate.gate_type())
                # if gate_types not in self.all_possible_gates and (gate_types[1], gate_types[0]) not in self.all_possible_gates:
                #     return False
                if isinstance(controlled_gate,
                              ControlledGate) and not controlled_gate.control_qubit_index == gate_index:
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
