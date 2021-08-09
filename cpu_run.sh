#!/bin/bash

# This is a script for submitting DNN training jobs on the HPC3 cluster.

#SBATCH --job-name=test                           ## Name of the job.
#SBATCH -A kgreif                                 ## account to charge 
#SBATCH -p free                                   ## partition/queue name
#SBATCH --nodes=1                                 ## (-N) number of nodes to use
#SBATCH --mem=30G                                 ## Request 3GB of memory per task
#SBATCH --ntasks-per-node=10                      ## Launch 10 tasks on node to get enough memory
#SBATCH --error=./outfiles/test.err       ## error log file
#SBATCH --output=./outfiles/test.out      ## output file

# Start by printing out hostname and date of job
echo "Found a node, here's some info: "
hostname; date
echo "================================"

# Get homedir name for future use
homedir=$(pwd)

# Now make directories to store training info/plots if they don't exist yet
trdir="${homedir}/training/test/"
mkdir -p ${trdir}/plots
mkdir -p ${trdir}/checkpoints

# We want to run in training directory, so change to that directory and print contents
cd ${trdir}
pwd
echo "Contents of directory before run..."
ls -lrth

# Next build command to run python training script
command="python ${homedir}/train_model.py -N 5"

# Run command
echo "================================"
echo "Will run command ${command}"
echo "================================"
$command
echo -e "\n\nDone!"
