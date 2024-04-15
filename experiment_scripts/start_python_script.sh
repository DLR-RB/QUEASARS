#!/bin/bash
conda_env='qpoetry'
use_gpu='false'
cluster_cpu=''
partition_cpu=''
cluster_gpu=''
partition_gpu=''
cluster=$cluster_cpu
partition=$partition_cpu
nodes='1'
tasks='1'
cpus='8'
gpus='1'
memory='32gb'
time='01:00:00'

print_usage() {
    cat <<USAGE

    Usage: $0 [--env conda_environment] [--use_gpu] [--nodes num_nodes] [--tasks num_tasks_per_node] [--cpus num_cpus] [--gpus num_gpus] [--memory memory_per_cpu] [--time time] [--script_name] [--script_args]

USAGE
    exit 1
}

for arg in "$@"; do
    case $arg in
    --env) conda_env=$2; shift 2 ;;
    --use_gpu) use_gpu='true'; shift ;;
    --nodes) nodes=$2; shift 2 ;;
    --tasks) tasks=$2; shift 2 ;;
    --cpus) cpus=$2; shift 2 ;;
    --gpus) gpus=$2; shift 2 ;;
    --memory) memory=$2; shift 2 ;;
    --time) time=$2; shift 2 ;;
    --script_name) script=$2; shift 2;;
    --script_args) script_args=$2; shift 2;;
    --help) print_usage ; shift ;;
  esac
done

if [ -z "$script" ]; then
  print_usage
fi

source_path=$0
source_shortened=${source_path%/*}
if [ "${#source_path}" == "${#source_shortened}" ]; then
        cd ..
else
        cd $source_shortened
        cd ..
fi

if [ "${use_gpu,,}" == 'true' ]; then
        echo using gpu
        sbatch_args="--cluster=${cluster_gpu} --partition=${partition_gpu} --nodes=${nodes} --ntasks-per-node=${tasks} --cpus-per-task=${cpus} --gres=gpu:${gpus} --mem=${memory} --time=${time}"
else
        echo using cpu
        sbatch_args="--cluster=${cluster_cpu} --partition=${partition_cpu} --nodes=${nodes} --ntasks-per-node=${tasks} --cpus-per-task=${cpus} --mem=${memory} --time=${time}"
fi

job_id="$(squeue -u $USER | tail -1| awk '{print $1}')"
if ! [ "$job_id" == "JOBID" ]; then
        echo Reconnecting to Job "$job_id"
        srun --pty --jobid $job_id bash -i
else
        echo Starting new Job
        sbatch $sbatch_args experiment_scripts/_start_python_job.sh --env "$conda_env" --script_name "$script" --script_args "$script_args"
fi
