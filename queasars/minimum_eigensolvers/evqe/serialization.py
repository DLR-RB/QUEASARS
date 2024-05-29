# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from json import JSONEncoder, JSONDecoder
from typing import Any

from queasars.minimum_eigensolvers.evqe.quantum_circuit.serialization import (
    EVQECircuitLayerEncoder,
    EVQECircuitLayerDecoder,
)
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.individual import EVQEIndividual
from queasars.minimum_eigensolvers.evqe.evolutionary_algorithm.population import EVQEPopulation


class EVQEPopulationJSONEncoder(JSONEncoder):
    """
    JSONEncoder class for encoding EVQEPopulation instances as JSON.
    This class can serialize the following QUEASARS classes:
        EVQEPopulation,
        EVQEIndividual,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._circuit_layer_encoder = EVQECircuitLayerEncoder(*args, **kwargs)

    def default(self, o: Any):

        if any(isinstance(o, t) for t in self._circuit_layer_encoder.serializable_types()):
            return self._circuit_layer_encoder.default(o)

        if isinstance(o, EVQEIndividual):
            return {
                "evqe_individual_n_qubits": o.n_qubits,
                "evqe_individual_layers": [self.default(layer) for layer in o.layers],
                "evqe_individual_parameter_values": list(o.parameter_values),
            }

        if isinstance(o, EVQEPopulation):
            if o.species_representatives is None:
                species_representatives = None
            else:
                species_representatives = [self.default(individual) for individual in o.species_representatives]

            if o.species_members is None:
                species_members = None
            else:
                species_members = [
                    [self.default(individual), members] for individual, members in o.species_members.items()
                ]

            if o.species_membership is None:
                species_membership = None
            else:
                species_membership = [
                    [individual_index, self.default(species_representative)]
                    for individual_index, species_representative in o.species_membership.items()
                ]

            return {
                "evqe_population_individuals": [self.default(individual) for individual in o.individuals],
                "evqe_population_species_representatives": species_representatives,
                "evqe_population_species_members": species_members,
                "evqe_population_species_membership": species_membership,
            }

    @staticmethod
    def serializable_types() -> set[type]:
        """
        :return: a set of all types, which this encoder can serialize
        :rtype: set[type]
        """
        return {
            EVQEIndividual,
            EVQEPopulation,
        }


class EVQEPopulationJSONDecoder(JSONDecoder):
    """
    JSONEncoder class for decoding EVQEPopulation instances from JSON.
    This class can deserialize the following QUEASARS classes:
        EVQEPopulation,
        EVQEIndividual,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)
        self._circuit_layer_decoder = EVQECircuitLayerDecoder(*args, **kwargs)

    @staticmethod
    def identifying_keys() -> set[str]:
        """
        :return: a set of all keys, which the object_hook of this decoder can identify
        :rtype: set[str]
        """

        return {
            "evqe_individual_n_qubits",
            "evqe_individual_layers",
            "evqe_individual_parameter_values",
            "evqe_population_individuals",
            "evqe_population_species_representatives",
            "evqe_population_species_members",
            "evqe_population_species_membership",
        }.union(EVQECircuitLayerDecoder.identifying_keys())

    def object_hook(self, object_dict):

        if any(key in self._circuit_layer_decoder.identifying_keys() for key in object_dict.keys()):
            return self._circuit_layer_decoder.object_hook(object_dict)

        if (
            "evqe_individual_n_qubits" in object_dict
            or "evqe_individual_layers" in object_dict
            or "evqe_individual_parameter_values" in object_dict
        ):
            return self.parse_individual(object_dict)

        if (
            "evqe_population_individuals" in object_dict
            or "evqe_population_species_representatives" in object_dict
            or "evqe_population_species_members" in object_dict
            or "evqe_population_species_membership" in object_dict
        ):
            return self.parse_population(object_dict)

    @staticmethod
    def parse_individual(object_dict):
        return EVQEIndividual(
            n_qubits=object_dict["evqe_individual_n_qubits"],
            layers=tuple(object_dict["evqe_individual_layers"]),
            parameter_values=tuple(object_dict["evqe_individual_parameter_values"]),
        )

    @staticmethod
    def parse_population(object_dict):
        individuals = tuple(object_dict["evqe_population_individuals"])
        species_representatives = object_dict["evqe_population_species_representatives"]

        species_members = object_dict["evqe_population_species_members"]
        if species_members is not None:
            species_members = dict(tuple(species_members))

        species_membership = object_dict["evqe_population_species_membership"]
        if species_membership is not None:
            species_membership = dict(species_membership)

        return EVQEPopulation(
            individuals=individuals,
            species_representatives=species_representatives,
            species_members=species_members,
            species_membership=species_membership,
        )
