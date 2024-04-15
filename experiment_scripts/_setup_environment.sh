#!/bin/bash

print_usage() {
    cat <<USAGE

    Usage: $0 [--env conda_environment]

USAGE
    exit 1
}

for arg in "$@"; do
    case $arg in
    --env) conda_env=$2; shift 2 ;;
    --help) print_usage ; shift ;;
  esac
done

if [ -z "$conda_env" ]; then
  print_usage
fi

module add miniconda3

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda config --remove channels defaults

conda config --add channels conda-forge

environment_match=$(conda env list | grep -w "$conda_env")
if [ -z "$environment_match" ]; then
    echo Environment not found. Creating environment!
    conda env create -n "$conda_env" --file conda_environment.yml
    conda activate "$conda_env"
    conda install gxx_linux-64 gcc_linux-64 swig -y
    conda install conda-forge::poetry -y
    poetry install
else
    echo Environment available!
    conda activate "$conda_env"
    poetry update
fi

path=$(realpath $0)
base_path="${path%$0}"
conda env config vars set PYTHONPATH="${base_path}"
conda deactivate
conda activate "$conda_env"

conda deactivate
