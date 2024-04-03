# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from json import JSONEncoder, JSONDecoder
from typing import Any

from queasars.job_shop_scheduling.problem_instances import (
    Machine,
    Operation,
    Job,
    JobShopSchedulingProblemInstance,
    UnscheduledOperation,
    ScheduledOperation,
    JobShopSchedulingResult,
)


class JSSPJSONEncoder(JSONEncoder):
    """
    JSONEncoder class for encoding objects of the job shop scheduling data classes in
    queasars.job_shop_scheduling.problem_instances
    This class can serialize:
        Machine,
        Operation,
        Job,
        JobShopSchedulingProblemInstance,
        ScheduledOperation,
        JobShopSchedulingResult,
    """

    def default(self, o: Any) -> Any:

        if isinstance(o, tuple):
            return {"tuple": [self.default(entry) for entry in o]}

        if isinstance(o, list):
            return [self.default(entry) for entry in o]

        if isinstance(o, dict):
            return {"dict": self.default(list(o.items()))}

        if isinstance(o, Machine):
            return {"machine_name": o.name}

        if isinstance(o, Operation):
            return {
                "operation_name": o.name,
                "operation_job_name": o.job_name,
                "operation_machine": self.default(o.machine),
                "operation_processing_duration": o.processing_duration,
            }

        if isinstance(o, Job):
            return {"job_name": o.name, "job_operations": self.default(o.operations)}

        if isinstance(o, JobShopSchedulingProblemInstance):
            return {
                "jssp_instance_name": o.name,
                "jssp_instance_machines": self.default(o.machines),
                "jssp_instance_jobs": self.default(o.jobs),
            }

        if isinstance(o, UnscheduledOperation):
            return {"unscheduled_operation": self.default(o.operation)}

        if isinstance(o, ScheduledOperation):
            return {
                "scheduled_operation": self.default(o.operation),
                "scheduled_start_time": self.default(o.start_time),
            }

        if isinstance(o, JobShopSchedulingResult):
            return {
                "jssp_result_problem_instance": self.default(o.problem_instance),
                "jssp_result_schedule": self.default(o.schedule),
            }

        return o


class JSSPJSONDecoder(JSONDecoder):
    """
    JSONDecoder class for decoding objects of the job shop scheduling data classes in
    queasars.job_shop_scheduling.problem_instances
    This class can decode:
        Machine,
        Operation,
        Job,
        JobShopSchedulingProblemInstance,
        ScheduledOperation,
        JobShopSchedulingResult,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, object_dict):

        if "tuple" in object_dict and len(object_dict) == 1:
            return self.parse_tuple(object_dict=object_dict)

        if "dict" in object_dict and len(object_dict) == 1:
            return self.parse_dict(object_dict=object_dict)

        if "machine_name" in object_dict:
            return self.parse_machine(object_dict=object_dict)

        if (
            "operation_name" in object_dict
            or "operation_job_name" in object_dict
            or "operation_machine" in object_dict
            or "operation_processing_duration" in object_dict
        ):
            return self.parse_operation(object_dict=object_dict)

        if "job_name" in object_dict or "job_operations" in object_dict:
            return self.parse_job(object_dict=object_dict)

        if (
            "jssp_instance_name" in object_dict
            or "jssp_instance_machines" in object_dict
            or "jssp_instance_jobs" in object_dict
        ):
            return self.parse_jssp_instance(object_dict=object_dict)

        if "unscheduled_operation" in object_dict:
            return self.parse_unscheduled_operation(object_dict=object_dict)

        if "scheduled_operation" in object_dict or "scheduled_start_time" in object_dict:
            return self.parse_scheduled_operation(object_dict=object_dict)

        if "jssp_result_problem_instance" in object_dict or "jssp_result_schedule" in object_dict:
            return self.parse_jssp_result(object_dict=object_dict)

    @staticmethod
    def parse_tuple(object_dict) -> tuple[Any]:
        return tuple(object_dict["tuple"])

    @staticmethod
    def parse_list(object_dict) -> list[Any]:
        return object_dict["list"]

    @staticmethod
    def parse_dict(object_dict) -> dict[Any, Any]:
        return dict(object_dict["dict"])

    @staticmethod
    def parse_machine(object_dict) -> Machine:
        return Machine(name=object_dict["machine_name"])

    @staticmethod
    def parse_operation(object_dict) -> Operation:
        return Operation(
            name=object_dict["operation_name"],
            job_name=object_dict["operation_job_name"],
            machine=object_dict["operation_machine"],
            processing_duration=object_dict["operation_processing_duration"],
        )

    @staticmethod
    def parse_job(object_dict) -> Job:
        return Job(
            name=object_dict["job_name"],
            operations=object_dict["job_operations"],
        )

    @staticmethod
    def parse_jssp_instance(object_dict) -> JobShopSchedulingProblemInstance:
        return JobShopSchedulingProblemInstance(
            name=object_dict["jssp_instance_name"],
            machines=object_dict["jssp_instance_machines"],
            jobs=object_dict["jssp_instance_jobs"],
        )

    @staticmethod
    def parse_unscheduled_operation(object_dict) -> UnscheduledOperation:
        return UnscheduledOperation(
            operation=object_dict["unscheduled_operation"],
        )

    @staticmethod
    def parse_scheduled_operation(object_dict) -> ScheduledOperation:
        return ScheduledOperation(
            operation=object_dict["scheduled_operation"],
            start_time=object_dict["scheduled_start_time"],
        )

    @staticmethod
    def parse_jssp_result(object_dict) -> JobShopSchedulingResult:
        return JobShopSchedulingResult(
            problem_instance=object_dict["jssp_result_problem_instance"],
            schedule=object_dict["jssp_result_schedule"],
        )
