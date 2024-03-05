# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

import pytest
from pytest import raises
from random import Random
from qiskit.circuit import QuantumCircuit

from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import (
    RotationGate,
    ControlGate,
    ControlledRotationGate,
    IdentityGate,
    EVQECircuitLayer,
    EVQECircuitLayerException,
    EVQEIndividual,
    EVQEIndividualException,
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
            assert (
                layer.is_valid()
            ), f"The {i}th sample circuit layer was invalid, even though all random circuit layers should be valid!"

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

            for j, gate in enumerate(next_layer.gates):
                if not isinstance(gate, IdentityGate):
                    assert gate != previous_layer.gates[j], (
                        f"For the {i}th sample, the next_layer added gates, "
                        + "which were already in the previous_layer!"
                    )

    def test_random_layer_seed(self):
        layer_0 = EVQECircuitLayer.random_layer(n_qubits=12, previous_layer=None, random_seed=0)
        assert layer_0 == EVQECircuitLayer.random_layer(
            n_qubits=12, previous_layer=None, random_seed=0
        ), "Different circuit layers were generated although the same seed was used!"

        layer_1 = EVQECircuitLayer.random_layer(n_qubits=12, previous_layer=layer_0, random_seed=0)
        assert layer_1 == EVQECircuitLayer.random_layer(
            n_qubits=12, previous_layer=layer_0, random_seed=0
        ), "Different circuit layers were generated although the same seed was used!"

    def test_parameterized_layer_gate_amount_of_parameters(self):
        circuit_layer = EVQECircuitLayer.random_layer(n_qubits=10, previous_layer=None, random_seed=0)
        parameterized_gate = circuit_layer.get_parameterized_layer_gate(layer_id=0)
        assert len(parameterized_gate.params) == circuit_layer.n_parameters, (
            "The amount of parameters specified by the circuit layer did "
            + "not match the amount of qiskit quantum gate parameters!"
        )

        circuit = QuantumCircuit(10)
        circuit.append(parameterized_gate, range(0, 10))
        assert len(circuit.parameters) == circuit_layer.n_parameters, (
            "The amount of parameters specified by the circuit layer did "
            + "not match the amount of qiskit quantum circuit parameters!"
        )

    def test_non_parameterized_layer_gate_amount_of_parameters(self):
        circuit_layer = EVQECircuitLayer.random_layer(n_qubits=10, previous_layer=None, random_seed=0)
        gate = circuit_layer.get_layer_gate(layer_id=0, parameter_values=(0,) * circuit_layer.n_parameters)
        assert (
            len(gate.params) == 0
        ), "The qiskit gate had unassigned parameters, although all parameter values should be assigned!"

        circuit = QuantumCircuit(10)
        circuit.append(gate, range(0, 10))
        assert (
            len(circuit.parameters) == 0
        ), "The qiskit circuit had unassigned parameters, although all parameter values should be assigned!"

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
                gate = RotationGate(qubit_index=qargs[0]._index)
                assert gate in filtered_gates, (
                    f"For the {i}th sample circuit layer, a rotation gate was found in the quantum circuit"
                    f"on qubit {gate.qubit_index} which was not expected there!"
                )
                filtered_gates.remove(gate)

            cu3_instructions = circuit.get_instructions("cu3")

            for _, qargs, _ in cu3_instructions:
                gate = ControlledRotationGate(qubit_index=qargs[1]._index, control_qubit_index=qargs[0]._index)
                assert gate in filtered_gates, (
                    f"For the {i}th sample circuit layer, a controlled rotation gate was found on in the quantum"
                    + f" circuit on the qubits {gate.qubit_index, gate.control_qubit_index} which was "
                    + f"not expected there!"
                )
                filtered_gates.remove(gate)

            assert len(filtered_gates) == 0, (
                f"For the {i}th sample circuit layer more quantum gates " + "were found than expected!"
            )


class TestEVQEIndividual:
    def test_random_individuals_are_valid(self):
        random_generator = Random(0)

        for _ in range(0, 100):
            individual = EVQEIndividual.random_individual(
                n_qubits=15,
                n_layers=10,
                randomize_parameter_values=False,
                random_seed=new_random_seed(random_generator),
            )
            assert (
                individual.is_valid()
            ), "An invalid individual was generated, although all random individuals should be valid!"

    def test_random_individual_seed(self):
        individual_1 = EVQEIndividual.random_individual(
            n_qubits=15,
            n_layers=10,
            randomize_parameter_values=False,
            random_seed=0,
        )
        assert individual_1 == EVQEIndividual.random_individual(
            n_qubits=15,
            n_layers=10,
            randomize_parameter_values=False,
            random_seed=0,
        ), "A different individual was generated for the same random seed!"

        individual_2 = EVQEIndividual.random_individual(
            n_qubits=20,
            n_layers=8,
            randomize_parameter_values=True,
            random_seed=0,
        )
        assert individual_2 == EVQEIndividual.random_individual(
            n_qubits=20,
            n_layers=8,
            randomize_parameter_values=True,
            random_seed=0,
        ), "A different individual was generated for the same random seed!"

    @pytest.fixture
    def individual(self) -> EVQEIndividual:
        return EVQEIndividual.random_individual(
            n_qubits=20, n_layers=10, randomize_parameter_values=True, random_seed=0
        )

    def test_change_parameter_values(self, individual):
        n_parameters = len(individual.get_parameter_values())
        new_parameters = (0,) * n_parameters

        new_individual = EVQEIndividual.change_parameter_values(individual=individual, parameter_values=new_parameters)

        assert (
            new_individual.n_qubits == individual.n_qubits
        ), "Changing the parameter values should not alter the amount of qubits of the individual!"
        assert (
            new_individual.layers == individual.layers
        ), "Changing the parameter values should not alter the amount of layers of the individual!"
        assert new_individual.parameter_values == new_parameters, "The parameter values were not changed successfully!"

    def test_change_layer_parameter_values(self, individual):
        layer_to_change = 4
        n_params_before = sum(layer.n_parameters for layer in individual.layers[0:layer_to_change])
        layer_n_params = individual.layers[layer_to_change].n_parameters

        new_parameters = (0,) * layer_n_params
        new_individual = EVQEIndividual.change_layer_parameter_values(
            individual=individual, layer_id=layer_to_change, parameter_values=new_parameters
        )

        assert (
            new_individual.n_qubits == individual.n_qubits
        ), "Changing the parameter values should not alter the amount of qubits of the individual!"
        assert (
            new_individual.layers == individual.layers
        ), "Changing the parameter values should not alter the amount of layers of the individual!"
        assert (
            new_individual.get_parameter_values()[0:n_params_before]
            == individual.get_parameter_values()[0:n_params_before]
        ), "The parameters of the unaffected layers should not be changed!"
        assert (
            new_individual.get_parameter_values()[n_params_before : n_params_before + layer_n_params] == new_parameters
        ), "The parameters of the affected layer were not changed successfully!"
        assert (
            new_individual.get_parameter_values()[n_params_before + layer_n_params :]
            == individual.get_parameter_values()[n_params_before + layer_n_params :]
        ), "The parameters of the unaffected layers should not be changed!"

    def test_add_0_random_layers(self, individual):
        with raises(EVQEIndividualException):
            EVQEIndividual.add_random_layers(
                individual=individual, n_layers=0, randomize_parameter_values=False, random_seed=0
            )

    def test_add_random_layers(self, individual):
        n_params_before = len(individual.get_parameter_values())
        n_new_random_layers = 3
        new_individual = EVQEIndividual.add_random_layers(
            individual=individual, n_layers=n_new_random_layers, randomize_parameter_values=False, random_seed=0
        )

        assert new_individual.is_valid()
        assert new_individual.n_qubits == individual.n_qubits, "Adding a layer to the individual made it invalid!"
        assert (
            len(new_individual.layers) == len(individual.layers) + n_new_random_layers
        ), "More or less layers than expected were added!"
        assert (
            new_individual.layers[: len(individual.layers)] == individual.layers
        ), "The previous layers should remain unaffected, but were somehow changed!"
        assert (
            new_individual.get_parameter_values()[0:n_params_before] == individual.get_parameter_values()
        ), "The parameters of the previous layers should remain unaffected, but were somehow changed!"

    def test_remove_0_layers(self, individual):
        with raises(EVQEIndividualException):
            EVQEIndividual.remove_layers(individual=individual, n_layers=0)

    def test_remove_all_layers(self, individual):
        with raises(EVQEIndividualException):
            EVQEIndividual.remove_layers(individual=individual, n_layers=len(individual.layers))

    def test_remove_layers(self, individual):
        n_layers_to_remove = 4
        n_params_upto_layer_4 = sum(layer.n_parameters for layer in individual.layers[0:-n_layers_to_remove])

        new_individual = EVQEIndividual.remove_layers(individual=individual, n_layers=n_layers_to_remove)
        assert (
            new_individual.n_qubits == individual.n_qubits
        ), "Removing layers changed the amount of qubits of the individual!"
        assert (
            new_individual.layers == individual.layers[0:-n_layers_to_remove]
        ), "The layers unaffected by the removal should remain the same, but were changed!"
        assert (
            new_individual.get_parameter_values() == individual.get_parameter_values()[0:n_params_upto_layer_4]
        ), "The parameter values of the layers unaffected by the removal should remain the same, but were changed!"

    def test_get_genetic_distance(self, individual):
        new_individual = EVQEIndividual.add_random_layers(
            individual=individual, n_layers=1, randomize_parameter_values=False, random_seed=0
        )
        assert (
            EVQEIndividual.get_genetic_distance(individual_1=individual, individual_2=new_individual) == 1
        ), "Adding a layer did not increase the genetic distance by 1 as expected!"

        new_individual = EVQEIndividual.remove_layers(individual=individual, n_layers=2)
        assert (
            EVQEIndividual.get_genetic_distance(individual_1=individual, individual_2=new_individual) == 1
        ), "Removing two layer did not decrease the genetic distance by 1 as expected!"

    def test_get_quantum_cirucit(self, individual):
        circuit = individual.get_quantum_circuit()
        n_prepended_gates = 1
        assert (
            circuit.depth() == len(individual.layers) + n_prepended_gates
        ), "The quantum circuit depth did not match the amount of circuit layers!"

    def test_get_parameterized_quantum_circuit(self, individual):
        circuit = individual.get_parameterized_quantum_circuit()
        n_prepended_gates = 1
        assert (
            circuit.depth() == len(individual.layers) + n_prepended_gates
        ), "The quantum circuit depth did not match the amount of circuit layers!"
        assert len(circuit.parameters) == len(
            individual.get_parameter_values()
        ), "The amount of parameters of the quantum circuit did not match the amount provided by the individual!"

    def test_get_partially_parameterized_quantum_circuit(self, individual):
        layer_ids = {2, 5}
        circuit = individual.get_partially_parameterized_quantum_circuit(layer_ids)
        n_prepended_gates = 1
        assert (
            circuit.depth() == len(individual.layers) + n_prepended_gates
        ), "The quantum circuit depth did not match the amount of circuit layers!"
        assert len(circuit.parameters) == individual.layers[2].n_parameters + individual.layers[5].n_parameters, (
            "The amount of parameters of the quantum circuit did not match the "
            + "amount of parameters of the specified layers!"
        )

    def test_get_parameters(self, individual):
        n_parameters = sum(layer.n_parameters for layer in individual.layers)
        assert len(individual.get_parameter_values()) == n_parameters, (
            "The amount of parameters provided by the individual does not match "
            + "the sum of the individual's layer's parameters!"
        )

    def test_get_layer_parameters(self, individual):
        assert (
            len(individual.get_layer_parameter_values(layer_id=7)) == individual.layers[7].n_parameters
        ), "get_layer_parameters did not return as many parameters as it should have!"

    def test_get_n_controlled_gates(self, individual):
        circuit = individual.get_quantum_circuit().decompose()
        assert (
            individual.get_n_controlled_gates() == circuit.count_ops()["cu3"]
        ), "get_n_controlled_gates() does not match the amount of control gates found in the quantum circuit!"
