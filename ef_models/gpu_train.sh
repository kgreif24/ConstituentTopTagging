#!/bin/bash

# This is a script for submitting EFN training jobs on the HPC3 cluster.
# It will use GPU accleration!

# Set up sbatch arguments

#SBATCH --job-name=trainEFN1Ms                    ## Name of the job.
#SBATCH -A kgreif                                 ## account to charge 
#SBATCH -p free-gpu                               ## partition/queue name
#SBATCH --gres=gpu:V100:1                         ## Use only 1 GPU
#SBATCH --nodes=1                                 ## (-N) number of nodes to use
#SBATCH --mem=26G                                 ## Request 26G of memory to fit larger data
#SBATCH --ntasks-per-node=9                       ## 26G // 3G gives you 9 tasks to launch on the node.

#SBATCH --array=1-1

#SBATCH --error=./outfiles/trainEFN1Ms_%j.err       ## error log file
#SBATCH --output=./outfiles/trainEFN1Ms_%j.out      ## output file

#SBATCH --mail-type=ALL                           ## Send email
#SBATCH --mail-user=kgreif@uci.edu                ## to this address

# Set up directory tree before sbatch arguments
homedir=$(pwd)
trdir="${homedir}/training/${SLURM_JOB_NAME}/run_${SLURM_ARRAY_TASK_ID}"
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

# We want output files to sit in trdir, so make that working directory
cd $trdir
echo "In directory ${trdir}"
ls -lrth

# Next build command to run python training script
command="python ${homedir}/train_efn.py -N 150 --nodes 60 --latent 64 -o ./checkpoints"

# Run command
echo "================================"
echo "Will run command ${command}"
echo "================================"
$command
echo -e "\n\nDone!"
