#!/bin/bash 
# SLURM SUBMIT SCRIPT
#SBATCH -N 2                     # Number of nodes
#SBATCH --gres=gpu:a100:2        # Number of GPUs per node
#SBATCH --ntasks-per-node=2      # Number of tasks per node (matches number of GPUs)
#SBATCH -c 32                    # Number of CPU cores allocated per task
#SBATCH --mem=16G                 # Memory per node
#SBATCH --time=0-00:10:00        # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm-%j.out    # Standard output and error log


# Load necessary modules
source $STORE/mypython/bin/activate

# Set NCCL parameters for multi-node communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

# Run the training script using srun
srun python training_script.py
