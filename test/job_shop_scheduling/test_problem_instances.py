# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2024 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.

from pytest import raises, fixture

from queasars.job_shop_scheduling.problem_instances import (
    Machine,
    Operation,
    Job,
    JobShopSchedulingProblemInstance,
    ScheduledOperation,
    JobShopSchedulingResult,
    JobShopSchedulingProblemException,
)


class TestMachines:

    def test_machine_initialization(self):
        Machine("test")

    def test_empty_name_forbidden(self):
        with raises(JobShopSchedulingProblemException):
            Machine("")


class TestOperation:

    def test_operation_initialization(self):
        Operation(name="test", job_name="test", machine=Machine("test"), processing_duration=1)

    def test_empty_name_forbidden(self):
        with raises(JobShopSchedulingProblemException):
            Operation(name="", job_name="test", machine=Machine("test"), processing_duration=5)
        with raises(JobShopSchedulingProblemException):
            Operation(name="test", job_name="", machine=Machine("test"), processing_duration=2)

    def test_invalid_makespan(self):
        with raises(JobShopSchedulingProblemException):
            Operation(name="test", job_name="test", machine=Machine("test"), processing_duration=0)
        with raises(JobShopSchedulingProblemException):
            Operation(name="test", job_name="test", machine=Machine("test"), processing_duration=-5)

    def test_equal_identifier(self):
        op_1 = Operation(name="op1", job_name="j1", machine=Machine("m1"), processing_duration=3)
        op_2 = Operation(name="op1", job_name="j1", machine=Machine("m2"), processing_duration=3)
        op_3 = Operation(name="op1", job_name="j1", machine=Machine("m1"), processing_duration=4)

        assert (
            op_1.identifier == op_2.identifier
        ), "Operations with the same name and job_name must have the same identifier!"
        assert (
            op_1.identifier == op_3.identifier
        ), "Operations with the same name and job_name must have the same identifier!"
        assert (
            op_2.identifier == op_3.identifier
        ), "Operations with the same name and job_name must have the same identifier!"

    def test_different_identifier(self):
        op_1 = Operation(name="op1", job_name="j1", machine=Machine("m1"), processing_duration=3)
        op_2 = Operation(name="op2", job_name="j1", machine=Machine("m1"), processing_duration=3)
        op_3 = Operation(name="op1", job_name="j2", machine=Machine("m1"), processing_duration=3)

        assert (
            op_1.identifier != op_2.identifier
        ), "Operations with differing name or job_name must not have the same identifier!"
        assert (
            op_1.identifier != op_3.identifier
        ), "Operations with differing name or job_name must not have the same identifier!"
        assert (
            op_2.identifier != op_3.identifier
        ), "Operations with differing name or job_name must not have the same identifier!"


class TestJob:

    def test_job_initialization(self):
        m1 = Machine("m1")
        m2 = Machine("m2")

        op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=1)
        op2 = Operation(name="op2", job_name="j1", machine=m2, processing_duration=2)

        Job(name="j1", operations=(op1, op2))

    def test_empty_name_forbidden(self):
        m1 = Machine("m1")
        m2 = Machine("m2")

        op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=1)
        op2 = Operation(name="op2", job_name="j1", machine=m2, processing_duration=2)

        with raises(JobShopSchedulingProblemException):
            Job(name="", operations=(op1, op2))

    def test_empty_operations_forbidden(self):
        with raises(JobShopSchedulingProblemException):
            Job(name="test", operations=tuple())

    def test_duplicate_operation_identifiers_forbidden(self):
        m1 = Machine("m1")
        m2 = Machine("m2")

        op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=1)
        op2 = Operation(name="op1", job_name="j1", machine=m2, processing_duration=2)

        with raises(JobShopSchedulingProblemException):
            Job(name="j1", operations=(op1, op2))

    def test_mismatched_job_name_forbidden(self):
        m1 = Machine("m1")
        m2 = Machine("m2")

        op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=1)
        op2 = Operation(name="op2", job_name="j2", machine=m2, processing_duration=2)

        with raises(JobShopSchedulingProblemException):
            Job(name="j1", operations=(op1, op2))

    def test_machine_revisiting_forbidden(self):
        m1 = Machine("m1")

        op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=1)
        op2 = Operation(name="op2", job_name="j1", machine=m1, processing_duration=2)

        with raises(JobShopSchedulingProblemException):
            Job(name="", operations=(op1, op2))

    def test_detect_wrong_machine(self):
        m1 = Machine("m1")
        m2 = Machine("m2")
        m3 = Machine("m3")

        op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=1)
        op2 = Operation(name="op2", job_name="j1", machine=m3, processing_duration=2)

        job = Job(name="j1", operations=(op1, op2))

        assert not job.is_consistent_with_machines(machines=(m1, m2))


class TestJobShopSchedulingProblemInstance:

    @staticmethod
    def make_single_op_job(job_name: str, machine: Machine) -> Job:
        op = Operation(name="op1", job_name=job_name, machine=machine, processing_duration=1)
        return Job(name=job_name, operations=(op,))

    def test_jssp_instance_job_initialization(self):
        m1 = Machine("m1")
        m2 = Machine("m2")

        j1 = self.make_single_op_job(job_name="j1", machine=m1)
        j2 = self.make_single_op_job(job_name="j2", machine=m2)

        JobShopSchedulingProblemInstance(name="instance", machines=(m1, m2), jobs=(j1, j2))

    def test_empty_name_forbidden(self):
        m1 = Machine("m1")
        m2 = Machine("m2")

        j1 = self.make_single_op_job(job_name="j1", machine=m1)
        j2 = self.make_single_op_job(job_name="j2", machine=m2)

        with raises(JobShopSchedulingProblemException):
            JobShopSchedulingProblemInstance(name="", machines=(m1, m2), jobs=(j1, j2))

    def test_non_distinct_machines_forbidden(self):
        m1 = Machine("m1")
        m2 = Machine("m1")

        j1 = self.make_single_op_job(job_name="j1", machine=m1)
        j2 = self.make_single_op_job(job_name="j2", machine=m2)

        with raises(JobShopSchedulingProblemException):
            JobShopSchedulingProblemInstance(name="instance", machines=(m1, m2), jobs=(j1, j2))

    def test_non_distinct_job_names_forbidden(self):
        m1 = Machine("m1")
        m2 = Machine("m2")

        j1 = self.make_single_op_job(job_name="j1", machine=m1)
        j2 = self.make_single_op_job(job_name="j1", machine=m2)

        with raises(JobShopSchedulingProblemException):
            JobShopSchedulingProblemInstance(name="instance", machines=(m1, m2), jobs=(j1, j2))

    def test_wrong_machine_usage_forbidden(self):
        m1 = Machine("m1")
        m2 = Machine("m2")
        m3 = Machine("m3")

        j1 = self.make_single_op_job(job_name="j1", machine=m1)
        j2 = self.make_single_op_job(job_name="j2", machine=m3)

        with raises(JobShopSchedulingProblemException):
            JobShopSchedulingProblemInstance(name="instance", machines=(m1, m2), jobs=(j1, j2))


class TestScheduledOperation:

    def test_valid_scheduled_operation(self):
        m = Machine("m")
        op = Operation(name="op", job_name="job", machine=m, processing_duration=2)
        scheduled_op = ScheduledOperation(operation=op, start=3)

        assert scheduled_op.is_scheduled, "If a start time is set, is_scheduled must be True!"
        assert (
            scheduled_op.start_time == 3
        ), "start_time must match the start specified in the scheduled operations constructor!"
        assert (
            scheduled_op.end_time == 5
        ), "end_time must match the start time plus the operation's processing duration!"

    def test_time_properties_of_invalid_scheduled_operation(self):
        m = Machine("m")
        op = Operation(name="op", job_name="job", machine=m, processing_duration=2)
        scheduled_op = ScheduledOperation(operation=op, start=None)
        assert not scheduled_op.is_scheduled, "If no start time is set, is_scheduled must be False!"
        with raises(JobShopSchedulingProblemException):
            time = scheduled_op.start_time
        with raises(JobShopSchedulingProblemException):
            time = scheduled_op.end_time


class TestJobShopSchedulingResult:

    @fixture
    def problem_instance(self) -> JobShopSchedulingProblemInstance:
        m1 = Machine("m1")
        m2 = Machine("m2")

        op1 = Operation(name="op1", job_name="j1", machine=m1, processing_duration=2)
        op2 = Operation(name="op2", job_name="j1", machine=m2, processing_duration=2)
        j1 = Job(name="j1", operations=(op1, op2))

        op3 = Operation(name="op3", job_name="j2", machine=m2, processing_duration=2)
        j2 = Job(name="j2", operations=(op3,))

        return JobShopSchedulingProblemInstance(name="instance", jobs=(j1, j2), machines=(m1, m2))

    def test_correct_result(self, problem_instance):
        schedule = {
            problem_instance.jobs[0]: (
                ScheduledOperation(operation=problem_instance.jobs[0].operations[0], start=0),
                ScheduledOperation(operation=problem_instance.jobs[0].operations[1], start=2),
            ),
            problem_instance.jobs[1]: (ScheduledOperation(operation=problem_instance.jobs[1].operations[0], start=0),),
        }
        result = JobShopSchedulingResult(problem_instance=problem_instance, schedule=schedule)

        assert result.is_valid, "For a valid schedule, the is_valid attribute must be true!"
        assert result.makespan == 4, "For the given schedule, the makespan must be 4!"

    def test_job_operation_overlap(self, problem_instance):
        schedule = {
            problem_instance.jobs[0]: (
                ScheduledOperation(operation=problem_instance.jobs[0].operations[0], start=2),
                ScheduledOperation(operation=problem_instance.jobs[0].operations[1], start=1),
            ),
            problem_instance.jobs[1]: (ScheduledOperation(operation=problem_instance.jobs[1].operations[0], start=3),),
        }
        result = JobShopSchedulingResult(problem_instance=problem_instance, schedule=schedule)

        assert not result.is_valid, "The given result must be invalid due to operation overlap in the first job!"
        assert result.makespan is None, "The makespan for an invalid result must be None!"

    def test_machine_operation_overlap(self, problem_instance):
        schedule = {
            problem_instance.jobs[0]: (
                ScheduledOperation(operation=problem_instance.jobs[0].operations[0], start=0),
                ScheduledOperation(operation=problem_instance.jobs[0].operations[1], start=2),
            ),
            problem_instance.jobs[1]: (ScheduledOperation(operation=problem_instance.jobs[1].operations[0], start=1),),
        }
        result = JobShopSchedulingResult(problem_instance=problem_instance, schedule=schedule)

        assert not result.is_valid, "The given result must be invalid due to operation overlap in the first job!"
        assert result.makespan is None, "The makespan for an invalid result must be None!"

    def test_unfitting_schedule_in_result(self, problem_instance):
        m1 = Machine("m1")
        op = Operation(name="op", job_name="j", machine=m1, processing_duration=2)
        j = Job(name="j", operations=(op,))
        sop = ScheduledOperation(operation=op, start=0)

        schedule = {j: (sop,)}
        with raises(JobShopSchedulingProblemException):
            JobShopSchedulingResult(problem_instance=problem_instance, schedule=schedule)
