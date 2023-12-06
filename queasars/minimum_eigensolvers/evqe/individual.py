# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Generator
from types import MappingProxyType
from random import choice, sample, random
from math import pi

from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.converters import circuit_to_gate

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BaseIndividual


class GateType(Enum):
    IDENTITY = 0
    ROTATION = 1
    CONTROLLED_ROTATION = 2


@dataclass(frozen=True, eq=True)
class CircuitLayer:
    n_qubits: int
    gate_assignments: tuple[GateType, ...]
    control_orders: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if not self.is_valid():
            raise CircuitLayerException("The created layer is invalid!")
        object.__setattr__(
            self,
            "_n_parameters",
            int(3 * (sum(1 for gate in self.gate_assignments if gate == GateType.ROTATION) + len(self.control_orders))),
        )

    @property
    def n_parameters(self) -> int:
        # This attribute is set in __post_init__ which mypy does not recognize. It is ensured to be an integer.
        return self._n_parameters  # type: ignore

    def is_valid(self) -> bool:
        is_valid = True

        if len(self.gate_assignments) != self.n_qubits:
            is_valid = False

        rotations = set()
        controls = set()

        for rotation, control in self.control_orders:
            if not (0 <= rotation < self.n_qubits and 0 <= control < self.n_qubits):
                is_valid = False
            if self.gate_assignments[rotation] != GateType.CONTROLLED_ROTATION:
                is_valid = False
            if rotation in rotations or rotation in controls:
                is_valid = False
            if self.gate_assignments[control] != GateType.CONTROLLED_ROTATION:
                is_valid = False
            if control in rotations or control in controls:
                is_valid = False

            rotations.add(rotation)
            controls.add(control)

        return is_valid

    def get_parameterized_layer_gate(self, layer_id: int) -> Gate:
        circuit = QuantumCircuit(self.n_qubits, name=f"Layer_{layer_id}")
        for qubit in range(0, self.n_qubits):
            gate = self.gate_assignments[qubit]

            if gate == GateType.IDENTITY:
                circuit.i(qubit)

            if gate == GateType.ROTATION:
                theta = Parameter(name=f"l{layer_id}_q{qubit}_theta")
                phi = Parameter(name=f"l{layer_id}_q{qubit}_phi")
                lam = Parameter(name=f"l{layer_id}_q{qubit}_lam")
                circuit.u(theta=theta, phi=phi, lam=lam, qubit=qubit)

        for rotation_qubit, control_qubit in self.control_orders:
            theta = Parameter(name=f"l{layer_id}_q{rotation_qubit}_theta")
            phi = Parameter(name=f"l{layer_id}_q{rotation_qubit}_phi")
            lam = Parameter(name=f"l{layer_id}_q{rotation_qubit}_lam")
            circuit.cu(
                theta=theta,
                phi=phi,
                lam=lam,
                gamma=0,
                control_qubit=control_qubit,
                target_qubit=rotation_qubit,
            )

        return circuit_to_gate(circuit)

    def get_qubit_parameter_index_mapping(self) -> dict[int, tuple[int, int, int]]:
        control_order_dict: dict[int, int] = dict(self.control_orders)
        parameter_index: int = 0
        result_mapping: dict[int, tuple[int, int, int]] = {}
        for qubit in range(0, self.n_qubits):
            if self.gate_assignments[qubit] == GateType.ROTATION or (
                self.gate_assignments[qubit] == GateType.CONTROLLED_ROTATION and qubit in control_order_dict
            ):
                result_mapping[qubit] = (
                    parameter_index,
                    parameter_index + 1,
                    parameter_index + 2,
                )
                parameter_index += 3
        return result_mapping

    @staticmethod
    def random_layer(n_qubits: int, previous_layer: Optional["CircuitLayer"] = None) -> "CircuitLayer":
        if previous_layer is not None and previous_layer.n_qubits != n_qubits:
            raise ValueError("previous_layer must have exactly as many qubits as the layer to be generated!")

        chosen_gate_types: list[GateType] = []
        controlled_qubits: list[int] = []
        control_orders: list[tuple[int, int]] = []

        for qubit in range(0, n_qubits):
            if previous_layer is not None and previous_layer.gate_assignments[qubit] in [
                GateType.ROTATION,
                GateType.IDENTITY,
            ]:
                chosen_gate_types.append(GateType.CONTROLLED_ROTATION)
                controlled_qubits.append(qubit)
            else:
                gate = choice([GateType.ROTATION, GateType.CONTROLLED_ROTATION])
                chosen_gate_types.append(gate)
                if gate == GateType.CONTROLLED_ROTATION:
                    controlled_qubits.append(qubit)

        while len(controlled_qubits) >= 2:
            rotation_qubit, control_qubit = sample(controlled_qubits, 2)
            if (
                previous_layer is not None and (rotation_qubit, control_qubit) not in previous_layer.control_orders
            ) or previous_layer is None:
                control_orders.append((rotation_qubit, control_qubit))
                controlled_qubits.remove(rotation_qubit)
                controlled_qubits.remove(control_qubit)

        if len(controlled_qubits) == 1:
            if (
                previous_layer is not None
                and previous_layer.gate_assignments[controlled_qubits[0]] == GateType.ROTATION
            ):
                chosen_gate_types[controlled_qubits[0]] = GateType.IDENTITY
            else:
                chosen_gate_types[controlled_qubits[0]] = GateType.ROTATION

        return CircuitLayer(
            n_qubits=n_qubits,
            gate_assignments=tuple(chosen_gate_types),
            control_orders=tuple(control_orders),
        )

    @staticmethod
    def squeeze_layers(
        left_layer: "CircuitLayer", right_layer: "CircuitLayer"
    ) -> Optional[tuple["CircuitLayer", dict[int, int]]]:
        if left_layer.n_qubits != right_layer.n_qubits:
            raise CircuitLayerException("Only layers with a matching amount of qubits can be squeezed!")
        left_parameter_mapping = left_layer.get_qubit_parameter_index_mapping()
        right_parameter_mapping = right_layer.get_qubit_parameter_index_mapping()
        new_gate_assignments: list[GateType] = list(left_layer.gate_assignments)
        index_mapping: dict[int, int] = dict()
        for qubit in range(0, left_layer.n_qubits):
            if not left_layer.gate_assignments[qubit] == GateType.ROTATION:
                new_gate_assignments[qubit] = left_layer.gate_assignments[qubit]
                continue

            if right_layer.gate_assignments[qubit] == GateType.ROTATION:
                new_gate_assignments[qubit] = GateType.IDENTITY
                for i in range(0, 3):
                    index_mapping[left_parameter_mapping[qubit][i]] = right_parameter_mapping[qubit][i]

        new_control_orders: list[tuple[int, int]] = list(left_layer.control_orders)
        for control_order in left_layer.control_orders:
            if control_order not in right_layer.control_orders:
                continue
            rotation_qubit, control_qubit = control_order
            new_gate_assignments[rotation_qubit] = GateType.IDENTITY
            new_gate_assignments[control_qubit] = GateType.IDENTITY
            new_control_orders.remove(control_order)
            for i in range(0, 3):
                index_mapping[left_parameter_mapping[rotation_qubit][i]] = right_parameter_mapping[rotation_qubit][i]

        if len(index_mapping) == 0:
            return None

        new_layer = CircuitLayer(
            n_qubits=left_layer.n_qubits,
            gate_assignments=tuple(new_gate_assignments),
            control_orders=tuple(new_control_orders),
        )

        return new_layer, index_mapping


class CircuitLayerException(Exception):
    pass


@dataclass(frozen=True, eq=True)
class EVQEIndividual(BaseIndividual):
    layers: tuple[CircuitLayer, ...]
    parameter_values: tuple[float, ...]

    @staticmethod
    def random_individual(n_qubits: int, n_layers: int) -> "EVQEIndividual":
        n_all_parameters: int = 0
        layers: list[CircuitLayer] = []
        layer: Optional[CircuitLayer] = None
        for _ in range(0, n_layers):
            layer = CircuitLayer.random_layer(n_qubits, layer)
            layers.append(layer)
            n_all_parameters += layer.n_parameters
        parameter_values: tuple[float, ...] = tuple((2 * pi * random() for _ in range(0, n_all_parameters)))
        return EVQEIndividual(layers=tuple(layers), parameter_values=parameter_values)

    @staticmethod
    def change_parameter_values(individual: "EVQEIndividual", parameter_values: tuple[float, ...]) -> "EVQEIndividual":
        return EVQEIndividual(layers=individual.layers, parameter_values=parameter_values)

    @staticmethod
    def change_layer_parameter_values(
        individual: "EVQEIndividual", layer_id: int, parameter_values: tuple[float, ...]
    ) -> "EVQEIndividual":
        parameter_layer_indices = individual.parameter_layer_indices[individual.layers[layer_id]]
        if not parameter_layer_indices[1] - parameter_layer_indices[0] == len(parameter_values):
            raise EVQEIndividualException("Number of parameters must match the specified layer!")
        new_parameter_values = list(individual.parameter_values)
        new_parameter_values[parameter_layer_indices[0] : parameter_layer_indices[1]] = parameter_values

        return EVQEIndividual(layers=individual.layers, parameter_values=tuple(new_parameter_values))

    @staticmethod
    def add_random_layers(individual: "EVQEIndividual", n_layers: int) -> "EVQEIndividual":
        new_layers: list[CircuitLayer] = []
        n_all_parameters: int = 0

        for _ in range(0, n_layers):
            layer = CircuitLayer.random_layer(individual.layers[0].n_qubits, previous_layer=individual.layers[-1])
            new_layers.append(layer)
            n_all_parameters += layer.n_parameters

        new_parameter_values: Generator[float, None, None] = (2 * pi * random() for _ in range(0, n_all_parameters))

        all_layers: tuple[CircuitLayer, ...] = (*individual.layers, *new_layers)
        all_parameter_values: tuple[float, ...] = (
            *individual.parameter_values,
            *new_parameter_values,
        )

        return EVQEIndividual(layers=all_layers, parameter_values=all_parameter_values)

    @staticmethod
    def remove_layer(individual: "EVQEIndividual", layer_id: int) -> "EVQEIndividual":
        layer_to_remove: CircuitLayer = individual.layers[layer_id]
        n_parameters: int = layer_to_remove.n_parameters

        if layer_id == 0:
            return EVQEIndividual(
                layers=individual.layers[1:],
                parameter_values=individual.parameter_values[n_parameters:],
            )
        if layer_id == len(individual.layers) - 1:
            return EVQEIndividual(
                layers=individual.layers[:-1],
                parameter_values=individual.parameter_values[:-n_parameters],
            )

        next_step: int = 2
        new_layers: list[CircuitLayer] = list(individual.layers[: layer_id - 1])
        current_parameter_values: list[float] = list(individual.parameter_values)
        parameter_values_to_remove: set[int] = set(
            range(
                individual.parameter_layer_indices[individual.layers[layer_id]][0],
                individual.parameter_layer_indices[individual.layers[layer_id]][1],
            )
        )

        for i in range(layer_id - 1, len(individual.layers) - 1):
            if i == layer_id:
                next_step = 1
                continue

            current_layer = individual.layers[i]
            next_layer = individual.layers[i + next_step]

            change = CircuitLayer.squeeze_layers(current_layer, next_layer)
            if change is None:
                new_layers.extend(individual.layers[i:-1])
                break

            patched_layer, index_mapping = change
            new_layers.append(patched_layer)

            for old_index, new_index in index_mapping.items():
                parameter_values_to_remove.add(old_index)
                current_parameter_values[
                    individual.parameter_layer_indices[next_layer][0] + new_index
                ] += current_parameter_values[individual.parameter_layer_indices[current_layer][0] + old_index]

        new_layers.append(individual.layers[-1])
        new_parameter_values: tuple[float, ...] = tuple(
            value for i, value in enumerate(current_parameter_values) if i not in parameter_values_to_remove
        )

        return EVQEIndividual(layers=tuple(new_layers), parameter_values=new_parameter_values)

    def __post_init__(self) -> None:
        if not self.is_valid():
            raise EVQEIndividualException("The created individual is not valid!")

        parameter_layer_indices: dict[CircuitLayer, tuple[int, int]] = {}
        parameter_index = 0
        for layer in self.layers:
            parameter_layer_indices[layer] = (
                parameter_index,
                parameter_index + layer.n_parameters,
            )
            parameter_index += layer.n_parameters

        object.__setattr__(self, "_parameter_layer_indices", MappingProxyType(parameter_layer_indices))

    @property
    def parameter_layer_indices(
        self,
    ) -> MappingProxyType[CircuitLayer, tuple[int, int]]:
        # It is initialized in __post_init__, which mypy does not recognize. Its typing is ensured there.
        return self._parameter_layer_indices  # type: ignore

    def is_valid(self) -> bool:
        is_valid = True

        if len(self.layers) == 0:
            is_valid = False

        n_qubits = self.layers[0].n_qubits
        for layer in self.layers:
            if not layer.is_valid():
                is_valid = False
            if not layer.n_qubits == n_qubits:
                is_valid = False

        if len(self.parameter_values) != sum(layer.n_parameters for layer in self.layers):
            is_valid = False

        return is_valid

    def get_quantum_circuit(self) -> QuantumCircuit:
        return self.get_partially_parameterized_quantum_circuit(set())

    def get_parameterized_quantum_circuit(self) -> QuantumCircuit:
        return self.get_partially_parameterized_quantum_circuit(parameterized_layers=set(range(0, len(self.layers))))

    def get_partially_parameterized_quantum_circuit(self, parameterized_layers: set[int]):
        n_qubits: int = self.layers[0].n_qubits
        circuit: QuantumCircuit = QuantumCircuit(n_qubits)
        for i, layer in enumerate(self.layers):
            gate = layer.get_parameterized_layer_gate(layer_id=i)
            circuit.append(instruction=gate, qargs=range(0, n_qubits))
            if i not in parameterized_layers:
                parameter_mapping = {
                    gate.params[i]: self.parameter_values[self.parameter_layer_indices[layer][0] + i]
                    for i in range(0, layer.n_parameters)
                }
                circuit.assign_parameters(parameters=parameter_mapping, inplace=True)
        return circuit

    def get_parameter_values(self) -> tuple[float, ...]:
        return self.parameter_values

    def __hash__(self):
        return hash((self.layers, self.parameter_values))


class EVQEIndividualException(Exception):
    pass
