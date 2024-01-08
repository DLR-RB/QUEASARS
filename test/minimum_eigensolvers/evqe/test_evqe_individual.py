# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from random import Random

from pytest import raises
from qiskit.circuit import QuantumCircuit

from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import (
    RotationGate,
    ControlGate,
    ControlledRotationGate,
    IdentityGate,
    EVQECircuitLayer,
    EVQECircuitLayerException,
)
from queasars.utility.random import new_random_seed


class TestEVQECircuitLayer:
    def test_too_few_gates_is_invalid(self):
        gates = (RotationGate(qubit_index=0), RotationGate(qubit_index=1))
        with raises(EVQECircuitLayerException):
            EVQECircuitLayer(n_qubits=3, gates=gates)

    def test_too_many_gates_is_invalid(self):
        gates = (RotationGate(qubit_index=0), RotationGate(qubit_index=1))
        with raises(EVQECircuitLayerException):
            EVQECircuitLayer(n_qubits=1, gates=gates)

    def test_wrong_gate_ordering_is_invalid(self):
        gates = (RotationGate(qubit_index=1), RotationGate(qubit_index=0))
        with raises(EVQECircuitLayerException):
            EVQECircuitLayer(n_qubits=2, gates=gates)

    def test_gate_index_out_of_bounds_is_invalid(self):
        gates = (RotationGate(-1), RotationGate(0), RotationGate(1))
        with raises(EVQECircuitLayerException):
            EVQECircuitLayer(n_qubits=3, gates=gates)

        gates = (RotationGate(1), RotationGate(2), RotationGate(3))
        with raises(EVQECircuitLayerException):
            EVQECircuitLayer(n_qubits=3, gates=gates)

    def test_loose_control_gate_is_invalid(self):
        gates = (
            ControlGate(qubit_index=0, controlled_qubit_index=2),
            ControlGate(qubit_index=1, controlled_qubit_index=2),
            ControlledRotationGate(qubit_index=2, control_qubit_index=1),
        )
        with raises(EVQECircuitLayerException):
            EVQECircuitLayer(n_qubits=3, gates=gates)

    def test_loose_controlled_rotation_gate_is_invalid(self):
        gates = (
            ControlledRotationGate(qubit_index=0, control_qubit_index=2),
            ControlledRotationGate(qubit_index=1, control_qubit_index=2),
            ControlGate(qubit_index=2, controlled_qubit_index=1),
        )
        with raises(EVQECircuitLayerException):
            EVQECircuitLayer(n_qubits=3, gates=gates)

    def test_random_layers_are_valid(self):
        random_generator = Random(0)
        for i in range(0, 100):
            layer = EVQECircuitLayer.random_layer(
                n_qubits=10, previous_layer=None, random_seed=new_random_seed(random_generator=random_generator)
            )
            assert layer.is_valid(), f"failed for random circuit layer number {i}"

    def test_random_layers_for_previous_layers_are_valid(self):
        random_generator = Random(0)
        for i in range(0, 100):
            previous_layer = EVQECircuitLayer.random_layer(
                n_qubits=10, previous_layer=None, random_seed=new_random_seed(random_generator=random_generator)
            )
            next_layer = EVQECircuitLayer.random_layer(
                n_qubits=10,
                previous_layer=previous_layer,
                random_seed=new_random_seed(random_generator=random_generator),
            )

            for i, gate in enumerate(next_layer.gates):
                if not isinstance(gate, IdentityGate):
                    assert gate != previous_layer.gates[i], f"invalid gate added in next_layer {i}"

    def test_parameterized_layer_gate_amount_of_parameters(self):
        circuit_layer = EVQECircuitLayer.random_layer(n_qubits=10, previous_layer=None, random_seed=0)
        parameterized_gate = circuit_layer.get_parameterized_layer_gate(layer_id=0)
        assert len(parameterized_gate.params) == circuit_layer.n_parameters

        circuit = QuantumCircuit(10)
        circuit.append(parameterized_gate, range(0, 10))
        assert len(circuit.parameters) == circuit_layer.n_parameters

    def test_non_parameterized_layer_gate_amount_of_parameters(self):
        circuit_layer = EVQECircuitLayer.random_layer(n_qubits=10, previous_layer=None, random_seed=0)
        parameterized_gate = circuit_layer.get_layer_gate(
            layer_id=0, parameter_values=(0,) * circuit_layer.n_parameters
        )
        assert len(parameterized_gate.params) == 0

        circuit = QuantumCircuit(10)
        circuit.append(parameterized_gate, range(0, 10))
        assert len(circuit.parameters) == 0

    def test_parameterized_layer_gate_circuit_for_all_needed_gates(self):
        random_generator = Random(0)
        for i in range(0, 100):
            circuit_layer = EVQECircuitLayer.random_layer(
                n_qubits=10, previous_layer=None, random_seed=new_random_seed(random_generator=random_generator)
            )
            parameterized_gate = circuit_layer.get_parameterized_layer_gate(layer_id=0)

            circuit = QuantumCircuit(10)
            circuit.append(parameterized_gate, range(0, 10))
            circuit = circuit.decompose()

            filtered_gates = [
                gate
                for gate in circuit_layer.gates
                if not isinstance(gate, IdentityGate) and not isinstance(gate, ControlGate)
            ]

            u_instructions = circuit.get_instructions("u")

            for _, qargs, _ in u_instructions:
                gate = RotationGate(qubit_index=qargs[0].index)
                assert gate in filtered_gates, f"failed for circuit layer {i}"
                filtered_gates.remove(gate)

            cu3_instructions = circuit.get_instructions("cu3")

            for _, qargs, _ in cu3_instructions:
                gate = ControlledRotationGate(qubit_index=qargs[1].index, control_qubit_index=qargs[0].index)
                assert gate in filtered_gates, f"failed for circuit layer {i}"
                filtered_gates.remove(gate)

            assert len(filtered_gates) == 0, f"failed for circuit layer {i}"
