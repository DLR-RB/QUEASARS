# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dataclasses import dataclass
from random import Random
from types import MappingProxyType
from typing import Optional
from math import pi, ceil

from qiskit.circuit import Gate, QuantumCircuit

from queasars.minimum_eigensolvers.base.evolutionary_algorithm import BaseIndividual
from queasars.minimum_eigensolvers.evqe.quantum_circuit.circuit_layer import EVQECircuitLayer
from queasars.utility.random import new_random_seed


@dataclass(frozen=True)
class EVQEIndividual(BaseIndividual):
    """Dataclass for individuals of the EVQE evolutionary algorithm, which
    represent a parameterized quantum circuit along with the corresponding parameter values

    :param n_qubits: Amount of qubits on which this individual's quantum circuit operates
    :type n_qubits: int
    :param layers: circuit layers of which this individual's quantum circuit consists of
    :type layers: tuple[EVQECircuitLayer, ...]
    :param parameter_values: parameter values with which to populate the parameterized rotation gates
    :type parameter_values: tuple[float, ...]
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
        :arg layer_id: index of the layer whose parameter values shall be changed. Negative indices are allowed.
            For instance -1 will refer to the last layer of the individual and -2 to the second to last layer
        :type layer_id: int
        :arg parameter_values: to set the layer's parameters to
        :type parameter_values: tuple[float, ...]
        :return: the new individual
        :rtype: EVQEIndividual
        """

        # loop indices to allow negative numbers to refer to the last layers of the individual
        layer_id = layer_id % len(individual.layers)

        if len(parameter_values) != len(individual.layer_parameter_indices[layer_id]):
            raise EVQEIndividualException(
                "The amount of given parameter_values does not match the amount needed by the circuit layer!"
            )

        layer_parameter_values: list[tuple[float, ...]] = []
        for index, layer in enumerate(individual.layers):
            if index != layer_id:
                layer_parameter_values.append(
                    tuple(individual.parameter_values[i] for i in individual.layer_parameter_indices[index])
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

        if n_layers < 1:
            raise EVQEIndividualException("n_layers must be at least 1!")

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
    def remove_layers(individual: "EVQEIndividual", n_layers: int) -> "EVQEIndividual":
        """Returns a new individual, based on the given individual, but with the last n_layers removed

        :arg individual: on which the new individual is based on
        :type individual: EVQEIndividual
        :arg n_layers: amount of last layers to remove
        :type n_layers: int
        :return: the new individual
        :rtype: EVQEIndividual
        """

        if not 0 < n_layers:
            raise EVQEIndividualException("n_layers must be at least 1!")
        if not n_layers < len(individual.layers):
            raise EVQEIndividualException(
                "Removed too many layers (one layer must remain)! Choose a smaller n_layer value"
            )

        # Remove the last layers
        layers: list[EVQECircuitLayer] = list(individual.layers)[0 : len(individual.layers) - n_layers]
        # Get the parameter values for the remaining layers
        parameter_values: list[float] = list(individual.parameter_values)[
            0 : individual.layer_parameter_indices[len(individual.layers) - n_layers][0]
        ]

        return EVQEIndividual(
            n_qubits=individual.n_qubits,
            layers=tuple(layers),
            parameter_values=tuple(parameter_values),
        )

    @staticmethod
    def get_genetic_distance(individual_1: "EVQEIndividual", individual_2: "EVQEIndividual") -> int:
        """
        Returns the genetic distance between two individuals, which is defined as the ceiling of the average amount of
        layers subtracted by the amount of shared layers

        :param individual_1: individual to determine genetic distance from
        :type individual_1: EVQEIndividual
        :param individual_2: individual to determine genetic distance from
        :type individual_2: EVQEIndividual
        :return:
        """
        n_layers_id1 = len(individual_1.layers)
        n_layers_id2 = len(individual_2.layers)
        n_all_layers: int = ceil(0.5 * (n_layers_id1 + n_layers_id2))
        n_shared_layers = 0
        for i in range(0, min(n_layers_id1, n_layers_id2)):
            if individual_1.layers[i] == individual_2.layers[i]:
                n_shared_layers += 1

        return n_all_layers - n_shared_layers

    def __post_init__(self) -> None:
        # Disallow the initialization of invalid individuals
        if not self.is_valid():
            raise EVQEIndividualException("The created individual is not valid!")

        # Initialize a mapping to hold the parameter indices for each layer
        layer_parameter_indices: dict[int, tuple[int, ...]] = {}
        parameter_index: int = 0
        for i, layer in enumerate(self.layers):
            layer_parameter_indices[i] = tuple(range(parameter_index, parameter_index + layer.n_parameters))
            parameter_index += layer.n_parameters
        object.__setattr__(self, "_layer_parameter_indices", MappingProxyType(layer_parameter_indices))

    def is_valid(self) -> bool:
        """Checks whether this individual is valid

        :return: True if the individual is valid, False otherwise
        :rtype: bool
        """

        # Check that the individual has at least one circuit layer
        if len(self.layers) <= 0:
            return False

        # Check that each layer is valid and of the correct size
        for layer in self.layers:
            if (not layer.is_valid()) or (not layer.n_qubits == self.n_qubits):
                return False

        # Check that the correct amount of parameters is provided
        if len(self.parameter_values) != sum(layer.n_parameters for layer in self.layers):
            return False

        return True

    @property
    def layer_parameter_indices(
        self,
    ) -> MappingProxyType[int, tuple[int, ...]]:
        """
        :return: A mapping, which maps the layer index to the parameter indices which belong ot it
        :rtype: MappingProxyType[EVQECircuitLayer, tuple[int, ...]]
        """
        # This attribute is set in __post_init__ which mypy does not recognize.
        return self._layer_parameter_indices  # type: ignore # pylint: disable=no-member

    def get_parameterized_quantum_circuit(self) -> QuantumCircuit:
        return self.get_partially_parameterized_quantum_circuit(set(range(0, len(self.layers))))

    def get_partially_parameterized_quantum_circuit(self, parameterized_layers: set[int]):
        """Returns the quantum circuit as represented by this individual with only some
        circuit layers being parameterized

        :arg parameterized_layers: indices of the circuit layers which shall be parameterized. The indices can also be
            negative to index layers starting from the last layer. For instance -1 refers to the last layer and -2 to
            the second to last layer of the individual
        :type parameterized_layers: set[int]
        :return: the partially parameterized QuantumCircuit
        :rtype: QuantumCircuit
        """
        n_qubits: int = self.layers[0].n_qubits
        circuit: QuantumCircuit = QuantumCircuit(n_qubits)

        # loop indices to allow negative numbers to refer to the last layers of the individual
        parameterized_layers = {layer_id % len(self.layers) for layer_id in parameterized_layers}

        # Apply each circuit layer one by one
        for i, layer in enumerate(self.layers):
            gate: Gate
            # Get the layer as parameterized gate, if required
            if i in parameterized_layers:
                gate = layer.get_parameterized_layer_gate(layer_id=i)
            else:
                layer_parameter_values: tuple[float, ...] = tuple(
                    parameter_value
                    for j, parameter_value in enumerate(self.parameter_values)
                    if j in self.layer_parameter_indices[i]
                )
                gate = layer.get_layer_gate(layer_id=i, parameter_values=layer_parameter_values)
            circuit.append(instruction=gate, qargs=range(0, n_qubits))

        circuit = circuit.decompose()

        return circuit

    def get_parameter_values(self) -> tuple[float, ...]:
        return self.parameter_values

    def get_layer_parameter_values(self, layer_id: int) -> tuple[float, ...]:
        """
        Returns the parameter values for a specified circuit layer

        :param layer_id: index of the layer. This value can also be negative to index layers starting from the last
            layer. For instance -1 refers to the last layer and -2 to the second to last layer of the individual
        :type layer_id: int
        :return: the parameter values
        :rtype: tuple[float, ...]
        """
        # loop indices to allow negative numbers to refer to the last layers of the individual
        layer_id = layer_id % len(self.layers)

        layer_parameter_values: tuple[float, ...] = tuple(
            parameter_value
            for i, parameter_value in enumerate(self.parameter_values)
            if i in self.layer_parameter_indices[layer_id]
        )
        return layer_parameter_values

    def get_n_controlled_gates(self) -> int:
        """Get the amount of controlled gates over all circuit layers of this individual

        :return: the number of controlled gates
        :rtype: int
        """
        return sum(layer.n_controlled_gates for layer in self.layers)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.n_qubits, self.layers, self.parameter_values))


class EVQEIndividualException(Exception):
    """Class for exceptions caused during operations on EVQEIndividuals"""
