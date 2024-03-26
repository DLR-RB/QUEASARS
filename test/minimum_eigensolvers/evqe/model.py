# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from docplex.mp.model import Model
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization.translators import from_docplex_mp, to_ising
from qiskit_optimization.converters import IntegerToBinary


def create_sample_model() -> Model:
    model = Model()
    x = model.integer_var(lb=0, ub=3, name="x")
    y = model.integer_var(lb=0, ub=3, name="y")
    model.minimize(x**2 - y**2)
    return model


def translate_model_to_hamiltonian(model: Model) -> tuple[SparsePauliOp, IntegerToBinary]:
    quadratic_program = from_docplex_mp(model=model)
    integer_converter = IntegerToBinary()
    quadratic_program = integer_converter.convert(problem=quadratic_program)
    hamiltonian, _ = to_ising(quad_prog=quadratic_program)
    return hamiltonian, integer_converter


def parse_bitstring(bitstring: str, converter: IntegerToBinary, reverse_bitstring: bool = True) -> list[int]:
    bitlist: list[int]
    if reverse_bitstring:
        bitlist = [int(char) for char in bitstring][::-1]
    else:
        bitlist = [int(char) for char in bitstring]
    return list(converter.interpret(bitlist))
