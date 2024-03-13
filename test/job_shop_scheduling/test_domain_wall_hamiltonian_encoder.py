# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from pytest import fixture, raises

from qiskit_algorithms.minimum_eigensolvers.diagonal_estimator import _evaluate_sparsepauli

from queasars.job_shop_scheduling.domain_wall_hamiltonian_encoder import JSSPDomainWallHamiltonianEncoder
from queasars.job_shop_scheduling.problem_instances import Machine, Operation, Job, JobShopSchedulingProblemInstance


class TestDomainWallHamiltonianEncoder:

    @fixture
    def problem_instance(self) -> JobShopSchedulingProblemInstance:
        m1 = Machine("m1")
        m2 = Machine("m2")

        op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=1)
        op2 = Operation(name="op2", job_name="j1", machine=m2, processing_duration=1)
        j1 = Job(name="j1", operations=(op1, op2))

        op3 = Operation(name="op3", job_name="j2", machine=m2, processing_duration=1)
        op4 = Operation(name="op4", job_name="j2", machine=m1, processing_duration=1)
        j2 = Job(name="j2", operations=(op3, op4))

        return JobShopSchedulingProblemInstance(name="instance", jobs=(j1, j2), machines=(m1, m2))

    def test_raises_for_too_small_timelimit(self, problem_instance):
        with raises(ValueError):
            encoder = JSSPDomainWallHamiltonianEncoder(jssp_instance=problem_instance, time_limit=1)
            encoder.get_problem_hamiltonian()

    def test_n_qubits(self, problem_instance):
        encoder = JSSPDomainWallHamiltonianEncoder(jssp_instance=problem_instance, time_limit=3)
        hamiltonian = encoder.get_problem_hamiltonian()
        assert (
            encoder.n_qubits == hamiltonian.num_qubits
        ), "The number of qubits reported by the JSSPDomainWallHamiltonianEncoder and the hamiltonian should match!"

    def test_encoding_constraint_energy_level(self, problem_instance):
        penalty = 100
        encoder = JSSPDomainWallHamiltonianEncoder(
            jssp_instance=problem_instance,
            time_limit=4,
            encoding_penalty=penalty,
            constraint_penalty=0,
            max_opt_value=0,
        )
        hamiltonian = encoder.get_problem_hamiltonian()
        n_qubits = encoder.n_qubits

        for i in range(0, 2**n_qubits):
            bitstring = format(i, f"0{n_qubits}b")
            result = encoder.translate_result_bitstring(bitstring=bitstring)
            energy_value = _evaluate_sparsepauli(state=i, observable=hamiltonian).real

            for job in problem_instance.jobs:
                if any(not operation.is_scheduled for operation in result.schedule[job]):
                    assert energy_value >= penalty, (
                        f"The state {i} was found to have an energy lower than the "
                        + f"penalty of {penalty} for an invalid encoding state!"
                    )

    def test_jssp_constraint_energy_level(self, problem_instance):
        penalty = 100
        encoder = JSSPDomainWallHamiltonianEncoder(
            jssp_instance=problem_instance,
            time_limit=4,
            encoding_penalty=0,
            constraint_penalty=penalty,
            max_opt_value=0,
        )
        hamiltonian = encoder.get_problem_hamiltonian()
        n_qubits = encoder.n_qubits

        for i in range(0, 2**n_qubits):
            bitstring = format(i, f"0{n_qubits}b")
            result = encoder.translate_result_bitstring(bitstring=bitstring)
            energy_value = _evaluate_sparsepauli(state=i, observable=hamiltonian).real

            encoding_violated = False
            for job in problem_instance.jobs:
                if any(not operation.is_scheduled for operation in result.schedule[job]):
                    encoding_violated = True
                    break

            if (not result.is_valid) and (not encoding_violated):
                assert energy_value >= penalty, (
                    f"The state {i} was found to have an energy lower than the "
                    + f"penalty of {penalty} for a state which violates the JSSP constraints!"
                )

    def test_optimization_energy_level(self, problem_instance):
        optimization_value = 100
        encoder = JSSPDomainWallHamiltonianEncoder(
            jssp_instance=problem_instance,
            time_limit=4,
            encoding_penalty=0,
            constraint_penalty=0,
            max_opt_value=optimization_value,
            opt_all_operations_share=0,
        )
        hamiltonian = encoder.get_problem_hamiltonian()
        n_qubits = encoder.n_qubits

        energy_values_per_makespan = {
            2: [],
            3: [],
            4: [],
        }
        for i in range(0, 2**n_qubits):
            bitstring = format(i, f"0{n_qubits}b")
            result = encoder.translate_result_bitstring(bitstring=bitstring)
            energy_value = _evaluate_sparsepauli(state=i, observable=hamiltonian).real

            if result.is_valid:
                assert energy_value <= optimization_value, (
                    f"The state {i} was found to have an energy higher than the "
                    + f"maximum optimization value of {optimization_value} for a valid result!"
                )
                energy_values_per_makespan[result.makespan].append(energy_value)

        assert max(energy_values_per_makespan[2]) < min(energy_values_per_makespan[3]), (
            "The maximum energy of solutions with makespan 2 is "
            + "higher than the minimum energy of solutions with makespan 3!"
        )
        assert max(energy_values_per_makespan[3]) < min(energy_values_per_makespan[4]), (
            "The maximum energy of solutions with makespan 3 is "
            + "higher than the minimum energy of solutions with makespan 4!"
        )
