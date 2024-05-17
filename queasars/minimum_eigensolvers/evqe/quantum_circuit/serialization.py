# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from json import JSONEncoder, JSONDecoder
from typing import Any

from queasars.minimum_eigensolvers.evqe.quantum_circuit.circuit_layer import EVQECircuitLayer
from queasars.minimum_eigensolvers.evqe.quantum_circuit.quantum_gate import (
    IdentityGate,
    RotationGate,
    ControlGate,
    ControlledRotationGate,
)


class EVQECircuitLayerEncoder(JSONEncoder):
    """
    JSONEncoder class for encoding EVQECircuitLayer instances as JSON.
    This class can serialize the following QUEASARS classes:
        EVQECircuitLayer,
        IdentityGate,
        RotationGate,
        ControlGate,
        ControlledRotationGate,
    """

    def default(self, o: Any):

        if isinstance(o, EVQECircuitLayer):
            return {
                "evqe_circuit_layer_n_qubits": o.n_qubits,
                "evqe_circuit_layer_gates": [self.default(gate) for gate in o.gates],
            }

        if isinstance(o, IdentityGate):
            return {
                "evqe_gate_type": "identity",
                "evqe_qubit_index": o.qubit_index,
            }

        if isinstance(o, RotationGate):
            return {
                "evqe_gate_type": "rotation",
                "evqe_qubit_index": o.qubit_index,
            }

        if isinstance(o, ControlGate):
            return {
                "evqe_gate_type": "control",
                "evqe_qubit_index": o.qubit_index,
                "evqe_controlled_qubit_index": o.controlled_qubit_index,
            }

        if isinstance(o, ControlledRotationGate):
            return {
                "evqe_gate_type": "controlled_rotation",
                "evqe_qubit_index": o.qubit_index,
                "evqe_control_qubit_index": o.control_qubit_index,
            }

    @staticmethod
    def serializable_types() -> set[type]:
        """
        :return: a set of all types, which this encoder can serialize
        :rtype: set[type]
        """
        return {
            EVQECircuitLayer,
            IdentityGate,
            RotationGate,
            ControlGate,
            ControlledRotationGate,
        }


class EVQECircuitLayerDecoder(JSONDecoder):
    """
    JSONDecoder class for decoding EVQECircuitLayer instances from JSON.
    This class can deserialize the following QUEASARS classes:
        EVQECircuitLayer,
        IdentityGate,
        RotationGate,
        ControlGate,
        ControlledRotationGate,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def identifying_keys() -> set[str]:
        """
        :return: a set of all keys, which the object_hook of this decoder can identify
        :rtype: set[str]
        """
        return {
            "evqe_circuit_layer_n_qubits",
            "evqe_circuit_layer_gates",
            "evqe_gate_type",
            "evqe_qubit_index",
            "evqe_controlled_qubit_index",
            "evqe_control_qubit_index",
        }

    def object_hook(self, object_dict):

        if "evqe_circuit_layer_n_qubits" in object_dict or "evqe_circuit_layer_gates" in object_dict:
            return self.parse_circuit_layer(object_dict)

        if "evqe_gate_type" in object_dict or "evqe_qubit_index" in object_dict:
            return self.parse_evqe_gate(object_dict)

    @staticmethod
    def parse_circuit_layer(object_dict):
        return EVQECircuitLayer(
            n_qubits=object_dict["evqe_circuit_layer_n_qubits"], gates=tuple(object_dict["evqe_circuit_layer_gates"])
        )

    @staticmethod
    def parse_evqe_gate(object_dict):
        gate_type = object_dict["evqe_gate_type"]

        if gate_type == "identity":
            return IdentityGate(qubit_index=object_dict["evqe_qubit_index"])

        if gate_type == "rotation":
            return RotationGate(qubit_index=object_dict["evqe_qubit_index"])

        if gate_type == "control":
            return ControlGate(
                qubit_index=object_dict["evqe_qubit_index"],
                controlled_qubit_index=object_dict["evqe_controlled_qubit_index"],
            )

        if gate_type == "controlled_rotation":
            return ControlledRotationGate(
                qubit_index=object_dict["evqe_qubit_index"], control_qubit_index=object_dict["evqe_control_qubit_index"]
            )

        raise ValueError(f"Encountered an unknown, serialized, evqe gate: {object_dict}!")
