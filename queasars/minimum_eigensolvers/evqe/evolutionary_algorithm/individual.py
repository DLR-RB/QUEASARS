# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from types import MappingProxyType
from random import Random
from math import pi, ceil

from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.circuit.library import CU3Gate
from qiskit.converters import circuit_to_gate

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BaseIndividual
from queasars.utility.random import new_random_seed


class EVQEGateType(Enum):
    """Enum representing the type of gate which can be placed for each qubit.
    This class is intended for structuring the genome and not for implementing quantum gate functionality
    """

    IDENTITY = 0
    ROTATION = 1
    CONTROL = 2
    CONTROLLED_ROTATION = 3


@dataclass(frozen=True, eq=True)
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


@dataclass(frozen=True, eq=True)
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
        circuit.i(self.qubit_index)


@dataclass(frozen=True, eq=True)
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


@dataclass(frozen=True, eq=True)
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


@dataclass(frozen=True, eq=True)
class ControlledGate(EVQEGate, ABC):
    """Dataclass representing a controlled quantum gate. It is not a valid gate on its own.
    It needs a matching ControlGate to be placed at the qubit of the control_qubit_index.
    This class is intended for structuring the genome and not for implementing quantum gate functionality

    :param control_qubit_index: The index of the qubit on which the belonging ControlGate is placed
    :type control_qubit_index: int
    """

    control_qubit_index: int


@dataclass(frozen=True, eq=True)
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


@dataclass(frozen=True, eq=True)
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

        # Ensure, that the previous layer matches the layer to be generated
        if previous_layer is not None and previous_layer.n_qubits != n_qubits:
            raise ValueError("previous_layer must have exactly as many qubits as the layer to be generated!")

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
            if (
                previous_layer is not None
                and previous_layer.gates[controlled_rotation_qubits[0]] == EVQEGateType.ROTATION
            ):
                chosen_gates[qubit_index] = IdentityGate(qubit_index=qubit_index)

        return EVQECircuitLayer(n_qubits=n_qubits, gates=tuple(chosen_gates))

    @staticmethod
    def squeeze_layers(
        left_layer: "EVQECircuitLayer", right_layer: "EVQECircuitLayer"
    ) -> tuple["EVQECircuitLayer", dict[int, int]]:
        """
        Tries to merge matching gates from the left_layer into the right_layer. This is done to
        remove unnecessary parameters. It returns a new left layer with the redundant gates removed,
        as well as a mapping of the parameter indices of removed gates in the left layer to the
        parameter indices of the respective gates in the right layer.

        :arg left_layer: left circuit layer to be squeezed
        :type left_layer: EVQECircuitLayer
        :arg right_layer: right circuit layer to be squeezed
        :type right_layer: EVQECircuitLayer
        :return: a tuple of the left layer with redundant gates removed and the parameter mapping
        :rtype: tuple[EVQECircuitLayer, dict[int, int]
        """

        # Ensure that the left and right layers fit together
        if left_layer.n_qubits != right_layer.n_qubits:
            raise EVQECircuitLayerException("Only layers with a matching amount of qubits can be squeezed!")

        # Initialize a buffer to hold the gates for the new layer
        left_gates: list[EVQEGate] = list(left_layer.gates)

        # Note which parameters on the old left layer map to which parameters on the right layer
        parameter_mapping: dict[int, int] = {}

        # Loop over all qubits and check for matching gates
        for qubit_index in range(0, left_layer.n_qubits):
            left_gate: EVQEGate = left_layer.gates[qubit_index]
            right_gate: EVQEGate = right_layer.gates[qubit_index]

            # If the gates match and are not identity gates, they can be squeezed
            if left_gate == right_gate and left_gate.gate_type() != EVQEGateType.IDENTITY:
                # Since the left gate is squeezed into the right layer,
                # remove it from the left layer
                left_gates[qubit_index] = IdentityGate(qubit_index=qubit_index)

                # Note the respective parameter mapping for the squeezed gates
                if left_gate.n_parameters() > 0:
                    left_parameter_indices = left_layer.get_qubit_parameter_mapping[qubit_index]
                    right_parameter_indices = right_layer.get_qubit_parameter_mapping[qubit_index]
                    for i in range(0, left_gate.n_parameters()):
                        parameter_mapping[left_parameter_indices[i]] = right_parameter_indices[i]

                continue

            # If the gates do not match, keep them as is
            left_gates[qubit_index] = left_gate

        new_left_layer = EVQECircuitLayer(
            n_qubits=left_layer.n_qubits,
            gates=tuple(left_gates),
        )

        return new_left_layer, parameter_mapping

    def __post_init__(self) -> None:
        # Buffer the amount of parameters offered by this circuit layer to prevent recalculating it
        object.__setattr__(self, "_n_parameters", int(sum(gate.n_parameters() for gate in self.gates)))

        # Disallow the creation of invalid circuit layers
        if not self.is_valid():
            raise EVQECircuitLayerException("The created layer is invalid!")

        # Buffer a mapping for the parameter values indices associated with each qubit
        parameter_index: int = 0
        result_mapping: dict[int, tuple[int, ...]] = {}
        for qubit_index, gate in enumerate(self.gates):
            if gate.n_parameters() > 0:
                result_mapping[qubit_index] = tuple(parameter_index + i for i in range(0, gate.n_parameters()))
                parameter_index += gate.n_parameters()
        object.__setattr__(self, "_qubit_parameter_index_mapping", MappingProxyType(result_mapping))

    @property
    def n_parameters(self) -> int:
        """
        :return: The number of parameters offered by this circuit layer
        :rtype: int
        """
        # This attribute is set in __post_init__ which mypy and pylint do not recognize.
        return self._n_parameters  # type: ignore # pylint: disable=no-member

    @property
    def get_qubit_parameter_mapping(self) -> MappingProxyType[int, tuple[int, ...]]:
        """
        Return a mapping from the qubit index to the indices of the parameter values associated with this qubit.
            If there are no parameters for this qubit, then there is no key - value pair for the qubits index.
            The returned mapping is immutable.

        :return: A mapping from the qubit index to the indices of the parameter values
        :rtype: MappingProxyType[int, tuple[int, ...]]
        """
        # This attribute is set in __post_init__ which mypy and pylint do not recognize.
        return self._qubit_parameter_index_mapping  # type: ignore # pylint: disable=no-member

    def is_valid(self) -> bool:
        """Checks whether this circuit layer is valid

        :return: True, if it is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # The circuit layer must have exactly as many gates as qubits
        if len(self.gates) != self.n_qubits:
            is_valid = False

        if not is_valid:
            return False

        # Check the validity of each gate
        for gate_index, gate in enumerate(self.gates):
            # Ensure each gate is at the right position in the gates tuple
            if gate_index != gate.qubit_index:
                is_valid = False
                break

            # Ensure each ControlledGate has a correctly assigned ControlGate
            if isinstance(gate, ControlledGate):
                control_gate = self.gates[gate.control_qubit_index]
                if not (isinstance(control_gate, ControlGate) and control_gate.controlled_qubit_index == gate_index):
                    is_valid = False
                    break

            # Ensure each ControlGate has a correctly assigned ControlledGate
            if isinstance(gate, ControlGate):
                controlled_gate = self.gates[gate.controlled_qubit_index]
                if not (
                    isinstance(controlled_gate, ControlledRotationGate)
                    and controlled_gate.control_qubit_index == gate_index
                ):
                    is_valid = False
                    break

        return is_valid

    def get_parameterized_layer_gate(self, layer_id: int) -> Gate:
        """Creates a parameterized qiskit Gate which can apply this circuit layer to any quantum circuit..

        :arg layer_id: id used in parameter and gate naming, should be unique for all layers of a quantum circuit
        :type layer_id: int
        :return: a qiskit Gate operator with which this layer can be applied
        :rtype: Gate
        """
        circuit = QuantumCircuit(self.n_qubits, name=f"layer_{layer_id}")

        layer_prefix = f"layer{layer_id}_"
        for gate in self.gates:
            gate.apply_gate(circuit=circuit, parameter_name_prefix=layer_prefix)

        return circuit_to_gate(circuit)

    def get_layer_gate(self, layer_id: int, parameter_values: tuple[float, ...]) -> Gate:
        """Creates a qiskit Gate which can apply this circuit layer to any quantum circuit using the given parameters.

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

        circuit = QuantumCircuit(self.n_qubits, name=f"layer_{layer_id}")

        layer_prefix = f"layer{layer_id}_"
        for gate in self.gates:
            gate.apply_gate(circuit=circuit, parameter_name_prefix=layer_prefix)

        circuit.assign_parameters(parameters=parameter_values, inplace=True)

        return circuit_to_gate(circuit)


class EVQECircuitLayerException(Exception):
    """Class for exceptions caused during operations involving the EVQECircuitLayer"""


@dataclass(frozen=True, eq=True)
class EVQEIndividual(BaseIndividual):
    """Dataclass for individuals of the EVQE evolutionary algorithm, which
    represent a parameterized quantum circuit along with the corresponding parameter values

    :param n_qubits: Amount of qubits on which this individual's quantum circuit operates
    :type n_qubits: int
    :param layers: circuit layers of which this individual's quantum circuit consists of
    :type layers: tuple[EVQECircuitLayer, ...]
    """

    n_qubits: int
    layers: tuple[EVQECircuitLayer, ...]
    parameter_values: tuple[float, ...]

    @staticmethod
    def random_individual(
        n_qubits: int, n_layers: int, randomize_parameter_values: bool, random_seed: Optional[int] = None
    ) -> "EVQEIndividual":
        """
        Creates a random individual for n_qubits with n_layers. Parameters can be initialized randomly
        or at 0.

        :arg n_qubits: amount of qubits on which the circuit of the generated individual shall act
        :type n_qubits: int
        :arg n_layers: amount of circuit layers in the generated individual's circuit
        :arg randomize_parameter_values: int
        :arg random_seed: integer value to control randomness
        :type random_seed: Optional[int]
        :return: the randomly generated individual
        :rtype: EVQEIndividual
        """
        random_generator: Random = Random(random_seed)
        layers: list[EVQECircuitLayer] = []
        layer: Optional[EVQECircuitLayer] = None
        for _ in range(0, n_layers):
            layer = EVQECircuitLayer.random_layer(
                n_qubits=n_qubits, previous_layer=layer, random_seed=new_random_seed(random_generator)
            )
            layers.append(layer)
        n_parameters: int = sum(layer.n_parameters for layer in layers)
        parameter_values: tuple[float, ...]
        if randomize_parameter_values:
            parameter_values = tuple(2 * pi * random_generator.random() for _ in range(0, n_parameters))
        else:
            parameter_values = (0,) * n_parameters
        return EVQEIndividual(n_qubits=n_qubits, layers=tuple(layers), parameter_values=parameter_values)

    @staticmethod
    def change_parameter_values(individual: "EVQEIndividual", parameter_values: tuple[float, ...]) -> "EVQEIndividual":
        """
        Returns a new individual with same circuit structure, but with changed parameter_values

        :arg individual: on which the new individual shall be based
        :type individual: EVQEIndividual
        :arg parameter_values: parameter_values to apply to the new individual
        :type parameter_values: tuple[float, ...]
        :return: the new individual
        :rtype: EVQEIndividual
        """
        if len(parameter_values) != sum(layer.n_parameters for layer in individual.layers):
            raise EVQEIndividualException("The number of parameter values given does not match the individual!")

        return EVQEIndividual(
            n_qubits=individual.n_qubits,
            layers=individual.layers,
            parameter_values=parameter_values,
        )

    @staticmethod
    def change_layer_parameter_values(
        individual: "EVQEIndividual", layer_id: int, parameter_values: tuple[float, ...]
    ) -> "EVQEIndividual":
        """Returns a new individual with the same circuit structure,
        but with changed parameter values for the specified circuit layer

        :arg individual: on which the new individual is based
        :type individual: EVQEIndividual
        :arg layer_id: index of the layer whose parameter values shall be changed
        :type layer_id: int
        :arg parameter_values: to set the layer's parameters to
        :type parameter_values: tuple[float, ...]
        :return: the new individual
        :rtype: EVQEIndividual
        """

        if not 0 <= layer_id < len(individual.layers):
            raise EVQEIndividualException("The given layer id is not valid!")

        if len(parameter_values) != len(individual.layer_parameter_indices[individual.layers[layer_id]]):
            raise EVQEIndividualException(
                "The amount of given parameter_values does not match the amount needed by the circuit layer!"
            )

        layer_parameter_values: list[tuple[float, ...]] = []
        for index, layer in enumerate(individual.layers):
            if index != layer_id:
                layer_parameter_values.append(
                    tuple(individual.parameter_values[i] for i in individual.layer_parameter_indices[layer])
                )
            else:
                layer_parameter_values.append(parameter_values)
        flattened_parameter_values: tuple[float, ...] = tuple(
            parameter_value for layer in layer_parameter_values for parameter_value in layer
        )

        return EVQEIndividual(
            n_qubits=individual.n_qubits,
            layers=individual.layers,
            parameter_values=flattened_parameter_values,
        )

    @staticmethod
    def add_random_layers(
        individual: "EVQEIndividual", n_layers: int, randomize_parameter_values: bool, random_seed: Optional[int] = None
    ) -> "EVQEIndividual":
        """Returns a new individual based on the given individual,
        but with additional random circuit layers appended. The parameter values for these
        layers can be initialized randomly or at 0.

        :arg individual: on which the new individual is based
        :type individual: EVQEIndividual
        :arg n_layers: amount of random circuit layers to append
        :type n_layers: int
        :arg randomize_parameter_values: whether to initialize the parameter values randomly
        :type randomize_parameter_values: bool
        :arg random_seed: integer value to control randomness
        :type random_seed: Optional[int]
        :return: the new individual
        :rtype: EVQEIndividual
        """
        new_layers: list[EVQECircuitLayer] = []
        random_generator = Random(random_seed)

        for _ in range(0, n_layers):
            layer = EVQECircuitLayer.random_layer(
                n_qubits=individual.layers[0].n_qubits,
                random_seed=new_random_seed(random_generator),
                previous_layer=individual.layers[-1],
            )
            new_layers.append(layer)

        all_layers: tuple[EVQECircuitLayer, ...] = (*individual.layers, *new_layers)

        n_new_parameters = sum(layer.n_parameters for layer in new_layers)
        new_parameter_values: tuple[float, ...]
        if randomize_parameter_values:
            new_parameter_values = tuple(2 * pi * random_generator.random() for _ in range(0, n_new_parameters))
        else:
            new_parameter_values = (0,) * n_new_parameters
        all_parameter_values: tuple[float, ...] = (
            *individual.parameter_values,
            *new_parameter_values,
        )

        return EVQEIndividual(
            n_qubits=individual.n_qubits,
            layers=all_layers,
            parameter_values=all_parameter_values,
        )

    @staticmethod
    def remove_layer(individual: "EVQEIndividual", layer_id: int) -> "EVQEIndividual":
        """Returns a new individual, based on the given individual, but with a specific
        circuit layer removed

        :param individual: on which the new individual is based on
        :type individual: EVQEIndividual
        :param layer_id: index of the layer to be removed
        :type layer_id: int
        :return: the new individual
        :rtype: EVQEIndividual
        """
        if not 0 <= layer_id < len(individual.layers):
            raise EVQEIndividualException("The given layer id is not valid!")

        # Remove the layer
        layers: list[EVQECircuitLayer] = list(individual.layers)
        layers.pop(layer_id)
        # Note the parameters of the removed layer to be removed
        parameters_to_remove: set[int] = set(
            parameter for parameter in individual.layer_parameter_indices[individual.layers[layer_id]]
        )

        adapted_parameter_values: list[float] = list(individual.parameter_values)

        # If the removed layer was neither the first nor the last, its surrounding layers can be squeezed
        if layer_id not in {0, len(individual.layers) - 1}:
            # Squeeze the surrounding layers
            left_layer = layers[layer_id - 1]
            right_layer = layers[layer_id]

            new_left_layer, parameter_mapping = EVQECircuitLayer.squeeze_layers(
                left_layer=left_layer, right_layer=right_layer
            )
            layers[layer_id - 1] = new_left_layer

            if len(parameter_mapping) > 0:
                left_starting_index: int = individual.layer_parameter_indices[left_layer][0]
                right_starting_index: int = individual.layer_parameter_indices[right_layer][0]

                # Add the parameter values of the removed gates on the left to the gates on the right
                for (
                    left_parameter_index,
                    right_parameter_index,
                ) in parameter_mapping.items():
                    adapted_parameter_values[left_starting_index + left_parameter_index] += individual.parameter_values[
                        right_starting_index + right_parameter_index
                    ]

                # Note the parameters of the squeezed gates to be removed
                parameters_to_remove.update(
                    left_starting_index + parameter_index for parameter_index in parameter_mapping
                )

        # Actually remove the parameters which were noted to be removed
        adapted_parameter_values = [
            parameter_value
            for i, parameter_value in enumerate(adapted_parameter_values)
            if i not in parameters_to_remove
        ]

        return EVQEIndividual(
            n_qubits=individual.n_qubits,
            layers=tuple(layers),
            parameter_values=tuple(adapted_parameter_values),
        )

    @staticmethod
    def get_genetic_distance(individual_1: "EVQEIndividual", individual_2: "EVQEIndividual") -> int:
        """
        Returns the genetic distance between two individuals, which is defined as the sum of the amount of
        all layers of both individuals subtracted by the amount of shared layers

        :param individual_1: individual to determine genetic distance from
        :type individual_1: EVQEIndividual
        :param individual_2: individual to determine genetic distance from
        :type individual_2: EVQEIndividual
        :return:
        """
        n_all_layers: int = ceil(0.5 * (len(individual_1.layers) + len(individual_2.layers)))
        n_shared_layers: int = len({layer for layer in individual_1.layers if layer in individual_2.layers})

        return n_all_layers - n_shared_layers

    def __post_init__(self) -> None:
        # Disallow the initialization of invalid individuals
        if not self.is_valid():
            raise EVQEIndividualException("The created individual is not valid!")

        # Initialize an immutable mapping to hold the parameter indices for each layer
        layer_parameter_indices: dict[EVQECircuitLayer, tuple[int, ...]] = {}
        parameter_index: int = 0
        for layer in self.layers:
            layer_parameter_indices[layer] = tuple(range(parameter_index, parameter_index + layer.n_parameters))
            parameter_index += layer.n_parameters
        object.__setattr__(self, "_layer_parameter_indices", MappingProxyType(layer_parameter_indices))

    def is_valid(self) -> bool:
        """Checks whether this individual is valid

        :return: True if the individual is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        # Check that the individual has at least one circuit layer
        if len(self.layers) <= 0:
            is_valid = False

        # Check that each layer is valid and of the correct size
        for layer in self.layers:
            if not layer.is_valid():
                is_valid = False
            if not layer.n_qubits == self.n_qubits:
                is_valid = False

        # Check that the correct amount of parameters is provided
        if len(self.parameter_values) != sum(layer.n_parameters for layer in self.layers):
            return False

        return is_valid

    @property
    def layer_parameter_indices(
        self,
    ) -> MappingProxyType[EVQECircuitLayer, tuple[int, ...]]:
        """
        :return: An immutable mapping, which maps the individuals circuit layers to
            the parameter indices which belong ot it
        :rtype: MappingProxyType[EVQECircuitLayer, tuple[int, ...]]
        """
        # This attribute is set in __post_init__ which mypy does not recognize.
        return self._layer_parameter_indices  # type: ignore # pylint: disable=no-member

    def get_quantum_circuit(self) -> QuantumCircuit:
        return self.get_partially_parameterized_quantum_circuit(set())

    def get_parameterized_quantum_circuit(self) -> QuantumCircuit:
        circuit: QuantumCircuit = self.get_quantum_circuit()
        circuit.assign_parameters(parameters=self.parameter_values, inplace=True)
        return circuit

    def get_partially_parameterized_quantum_circuit(self, parameterized_layers: set[int]):
        """Returns the quantum circuit as represented by this individual with only some
        circuit layers being parameterized

        :arg parameterized_layers: indices of the circuit layers which shall be parameterized
        :type parameterized_layers: set[int]
        :return: the partially parameterized QuantumCircuit
        :rtype: QuantumCircuit
        """
        n_qubits: int = self.layers[0].n_qubits
        circuit: QuantumCircuit = QuantumCircuit(n_qubits)

        # Apply each circuit layer one by one
        for i, layer in enumerate(self.layers):
            gate: Gate
            # Get the layer as parameterized gate, if required
            if i in parameterized_layers:
                gate = layer.get_parameterized_layer_gate(layer_id=i)
            else:
                layer_parameter_values: tuple[float, ...] = tuple(
                    parameter_value
                    for i, parameter_value in enumerate(self.parameter_values)
                    if i in self.layer_parameter_indices[layer]
                )
                gate = layer.get_layer_gate(layer_id=i, parameter_values=layer_parameter_values)
            circuit.append(instruction=gate, qargs=range(0, n_qubits))
            circuit.barrier()
        return circuit

    def get_parameter_values(self) -> tuple[float, ...]:
        return self.parameter_values

    def get_layer_parameter_values(self, layer_id: int) -> tuple[float, ...]:
        """
        Returns the parameter values for a specified circuit layer

        :param layer_id: index of the layer
        :type layer_id: int
        :return: the parameter values
        :rtype: tuple[float, ...]
        """
        layer_parameter_values: tuple[float, ...] = tuple(
            parameter_value
            for i, parameter_value in enumerate(self.parameter_values)
            if i in self.layer_parameter_indices[self.layers[layer_id]]
        )
        return layer_parameter_values

    def __hash__(self):
        return hash((self.n_qubits, self.layers))


class EVQEIndividualException(Exception):
    """Class for exceptions caused during operations on EVQEIndividuals"""
