#!/bin/bash

# This is a script for submitting DNN training jobs on the HPC3 cluster.

#SBATCH --job-name=trainDNN                       ## Name of the job.
#SBATCH -A kgreif                                 ## account to charge 
#SBATCH -p free                                   ## partition/queue name
#SBATCH --nodes=1                                 ## (-N) number of nodes to use
#SBATCH --ntasks=1                                ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=1                         ## number of cores the job needs
#SBATCH --mem=6G                                  ## Request 6GB of memory
#SBATCH --error=./dnn/training/trainDNN.err       ## error log file
#SBATCH --output=./dnn/training/trainDNN.out      ## output file

# Start by printing out hostname and date of job
echo "Found a node, here's some info: "
hostname; date
echo "================================"

# Load needed modules
module load pytorch/1.5.1

# Get homedir name for future use
homedir=$(pwd)

# Now make directories to store training info/plots if they don't exist yet
trdir="${homedir}/dnn/training/test_weights/"
mkdir -p ${trdir}/plots
mkdir -p ${trdir}/checkpoints

# We want output files to sit in trdir, so make that working directory
cd $trdir
echo "In directory ${trdir}"
ls -lrth

# Next build command to run python training script
command="python ${homedir}/train_dnn.py --nodes 100 -N 5"

# Run command
echo "================================"
echo "Will run command ${command}"
echo "================================"
$command
echo -e "\n\nDone!"
