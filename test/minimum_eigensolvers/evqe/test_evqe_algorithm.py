# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from dask.distributed import LocalCluster
import pytest

from test.minimum_eigensolvers.evqe.model import create_sample_model, translate_model_to_hamiltonian, parse_bitstring
from test.minimum_eigensolvers.evqe.solver import (
    create_sample_solver,
    get_likeliest_bitstrings_from_result,
)


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(n_workers=2)
    client = cluster.get_client()
    yield client
    cluster.close()


class TestEVQEAlgorithm:

    def test_simple_example(self, dask_client):
        model = create_sample_model()
        hamiltonian, integer_converter = translate_model_to_hamiltonian(model)

        solver = create_sample_solver(dask_client)
        result = solver.compute_minimum_eigenvalue(operator=hamiltonian)

        bitstring = get_likeliest_bitstrings_from_result(result=result)[0]

        variable_values = parse_bitstring(bitstring, integer_converter)

        # The global minimum of the sample model is [0, 3]. Should the sample model change,
        # these values have to be adjusted
        assert (
            variable_values[0] == 0 and variable_values[1] == 3
        ), f"The global minimum is [0, 3], but the solver found {variable_values}"
