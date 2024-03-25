# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from json import dumps, loads

from queasars.job_shop_scheduling.serialization import JSSPJSONEncoder, JSSPJSONDecoder
from test.job_shop_scheduling.problem_instance import problem_instance, valid_result, invalid_result


class TestSerialization:

    def test_serialize_deserialize_problem_instance(self):
        instance = problem_instance()
        serialized = dumps(obj=instance, cls=JSSPJSONEncoder, indent=2)
        deserialized = loads(s=serialized, cls=JSSPJSONDecoder)

        assert instance == deserialized, "Serializing and deserializing resulted in a different problem instance!"

    def test_serialize_deserialize_valid_result(self):
        result = valid_result()
        serialized = dumps(obj=result, cls=JSSPJSONEncoder, indent=2)
        deserialized = loads(s=serialized, cls=JSSPJSONDecoder)

        assert (
            result.problem_instance == deserialized.problem_instance
        ), "Serializing and deserializing changed the problem instance of the result!"
        assert (
            result.schedule == deserialized.schedule
        ), "Serializing and deserializing changed the schedule of the result!"

    def test_serialize_deserialize_invalid_result(self):
        result = invalid_result()
        serialized = dumps(obj=result, cls=JSSPJSONEncoder, indent=2)
        deserialized = loads(s=serialized, cls=JSSPJSONDecoder)

        assert (
            result.problem_instance == deserialized.problem_instance
        ), "Serializing and deserializing changed the problem instance of the result!"
        assert (
            result.schedule == deserialized.schedule
        ), "Serializing and deserializing changed the schedule of the result!"
