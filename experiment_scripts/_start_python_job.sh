#!/bin/bash
#SBATCH -J pythonscript_job
#SBATCH -o experiment_scripts/logs/%x.%j.out
#SBATCH -e experiment_scripts/logs/%x.%j.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=
#SBATCH --partition=
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --mail-type=all
#SBATCH --mail-user=
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --account=

print_usage() {
    cat <<USAGE

    Usage: $0 [--env conda_environment] [--script_name] [--script_args]

USAGE
    exit 1
}

module load slurm_setup

for arg in "$@"; do
    case $arg in
    --env) conda_env=$2; shift 2 ;;
    --script_name) script=$2; shift 2 ;;
    --script_args) script_args=$2; shift 2 ;;
  esac
done

if [ -z "$conda_env" ]; then
  print_usage
fi

if [ -z "$script" ]; then
  print_usage
fi

sh experiment_scripts/_setup_environment.sh --env "$conda_env"

module add miniconda3

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate "$conda_env"

poetry run python experiment_scripts/${script} ${script_args}
