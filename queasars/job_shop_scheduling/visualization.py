# Quantum Evolving Ansatz Variational Solver (QUEASARS)
# Copyright 2023 DLR - Deutsches Zentrum für Luft- und Raumfahrt e.V.
from typing import Any

from matplotlib import pyplot
from matplotlib import colormaps
from matplotlib import patches
from matplotlib.figure import Figure

from queasars.job_shop_scheduling.problem_instances import (
    JobShopSchedulingProblemInstance,
    JobShopSchedulingResult,
    Machine,
    Job,
    ScheduledOperation,
)


def plot_jssp_problem_instance_gantt(problem_instance: JobShopSchedulingProblemInstance):
    """
    Plots a job shop scheduling problem instance using matplotlib

    :arg problem_instance: to plot
    :type problem_instance: JobShopSchedulingProblemInstance
    """

    fig, ax = pyplot.subplots()

    cmap = colormaps["Set1"]
    machine_color_map: dict = {machine: cmap(i) for i, machine in enumerate(problem_instance.machines)}

    max_end = 0
    for i, job in enumerate(problem_instance.jobs):
        start = 0
        x_ranges = []
        colors = []
        for operation in job.operations:
            x_ranges.append((start + 0.02, operation.processing_duration - 0.04))
            colors.append(machine_color_map[operation.machine])
            start += operation.processing_duration
        ax.broken_barh(xranges=x_ranges, yrange=(i + 0.75, 0.5), color=colors)

        max_end = max(start, max_end)

    ax.set_title(f"Problem Instance: {problem_instance.name}")
    ax.set_yticks(range(1, len(problem_instance.jobs) + 1))
    ax.set_yticklabels(job.name for job in problem_instance.jobs)
    ax.set_ylabel("Jobs")
    ax.set_xticks(range(0, max_end + 1))
    ax.set_xlabel("Time")
    _create_color_legend(fig=fig, color_labels={color: machine.name for machine, color in machine_color_map.items()})

    pyplot.show()


def plot_jssp_problem_solution_gantt(result: JobShopSchedulingResult):
    """
    Plots a job shop scheduling result using matplotlib

    :arg result: to plot
    :type result: JobShopSchedulingResult
    """
    fig, ax = pyplot.subplots()

    cmap = colormaps["Set1"]
    job_color_map: dict = {job: cmap(i) for i, job in enumerate(result.problem_instance.jobs)}

    machine_schedule: dict[Machine, list[tuple[ScheduledOperation, Job]]] = {
        machine: [] for machine in result.problem_instance.machines
    }
    for job, scheduled_operations in result.schedule.items():
        for scheduled_operation in scheduled_operations:
            machine_schedule[scheduled_operation.operation.machine].append((scheduled_operation, job))

    for i, machine in enumerate(result.problem_instance.machines):
        x_ranges = []
        colors = []
        for scheduled_operation, job in machine_schedule[machine]:
            if scheduled_operation.schedule is not None:
                x_ranges.append(
                    (scheduled_operation.schedule[0] + 0.02, scheduled_operation.operation.processing_duration - 0.04)
                )
                colors.append(job_color_map[job])
        if len(x_ranges) > 0:
            ax.broken_barh(xranges=x_ranges, yrange=(i + 0.75, 0.5), color=colors)

    ax.set_title(f"Scheduling Result: {result.problem_instance.name}")
    ax.set_yticks(range(1, len(result.problem_instance.machines) + 1))
    ax.set_yticklabels(machine.name for machine in result.problem_instance.machines)
    ax.set_ylabel("Machines")
    ax.set_xticks(range(0, result.makespan + 1))
    ax.set_xlabel("Time")

    _create_color_legend(fig=fig, color_labels={color: job.name for job, color in job_color_map.items()})

    pyplot.show()


def _create_color_legend(fig: Figure, color_labels: dict[Any, str]) -> None:
    patch_list: list = []
    for color, label in color_labels.items():
        patch_list.append(patches.Patch(color=color, label=label))
    fig.legend(handles=patch_list, title="Machines", loc="center left", bbox_to_anchor=(0.9, 0.5))
