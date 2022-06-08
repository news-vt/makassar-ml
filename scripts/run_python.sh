#!/bin/bash
#SBATCH --partition=p100_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 4
#SBATCH --time=12:00:00
#SBATCH --account="ece6524-spring2022"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@vt.edu

# Load modules.
echo "Loading modules"
module reset >/dev/null 2>&1
module load Anaconda3 >/dev/null 2>&1

# Build anaconda environment name from SLURM allocation.
CONDA_ENV_NAME="tf-$(cut -d '_' -f 1 <<< $SLURM_JOB_PARTITION)"

# Initialize the shell to use Anaconda.
eval "$(conda shell.bash hook)"

# Activate Anaconda environment.
conda activate ${CONDA_ENV_NAME}

# Propagate arguments to Python and run.
python $@
