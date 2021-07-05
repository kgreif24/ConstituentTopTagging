#!/bin/bash

# This is a script for submitting DNN training jobs on the HPC3 cluster.
# It will use GPU accleration!

# Set up sbatch arguments

#SBATCH --job-name=trainDNN1Ms                    ## Name of the job.
#SBATCH -A kgreif                                 ## account to charge 
#SBATCH -p free-gpu                               ## partition/queue name
#SBATCH --gres=gpu:V100:1                         ## Use only 1 GPU
#SBATCH --nodes=1                                 ## (-N) number of nodes to use
#SBATCH --ntasks=1                                ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=1                         ## number of cores the job needs
#SBATCH --mem=6G                                  ## Request 6GB of memory

#SBATCH --array=1-5

#SBATCH --error=./outfiles/trainDNN1Ms_%j.err       ## error log file
#SBATCH --output=./outfiles/trainDNN1Ms_%j.out      ## output file

#SBATCH --mail-type=ALL                           ## Send email
#SBATCH --mail-user=kgreif@uci.edu                ## to this address

# Set up directory tree before sbatch arguments
homedir=$(pwd)
trdir="${homedir}/dnn/training/${SLURM_JOB_NAME}/run_${SLURM_ARRAY_TASK_ID}"
mkdir -p ${trdir}/plots
mkdir -p ${trdir}/checkpoints

# On to running the job!
# Start by printing out hostname and date of job
echo "Found a node, here's some info: "
hostname; date
echo $SLURM_JOB_NAME
echo $SLURM_JOB_ID
echo $SLURM_ARRAY_TASK_ID
echo "================================"

# Load needed modules
module purge
module load pytorch/1.5.1

# We want output files to sit in trdir, so make that working directory
cd $trdir
echo "In directory ${trdir}"
ls -lrth

# Next build command to run python training script
command="python ${homedir}/train_dnn.py --enableCuda -N 150 --nodes 50"

# Run command
echo "================================"
echo "Will run command ${command}"
echo "================================"
$command
echo -e "\n\nDone!"
