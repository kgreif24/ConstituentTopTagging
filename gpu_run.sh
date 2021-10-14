#!/bin/bash

# This is a script for submitting training jobs on the HPC3 cluster.
# It will use GPU accleration!

# Set up sbatch arguments

#SBATCH --job-name=test                           ## Name of the job.
#SBATCH -A kgreif                                 ## account to charge 
#SBATCH -p free-gpu                               ## partition/queue name
#SBATCH --gres=gpu:V100:1                         ## Use only 1 GPU
#SBATCH --nodes=1                                 ## (-N) number of nodes to use
#SBATCH --mem=30G                                 ## Request 30G of memory to fit data
#SBATCH --ntasks-per-node=10

#SBATCH --array=1-1

#SBATCH --error=./outfiles/%x_%a.err       ## error log file
#SBATCH --output=./outfiles/%x_%a.out      ## output file

#SBATCH --mail-type=END,FAIL                      ## Send email
#SBATCH --mail-user=kgreif@uci.edu                ## to this address

# Set up directory tree before sbatch arguments
homedir=$(pwd)
trdir="${homedir}/training/${SLURM_JOB_NAME}/run_${SLURM_ARRAY_TASK_ID}"
logdir="${homedir}/training/${SLURM_JOB_NAME}/logs"
rm -rf ${trdir}
mkdir -p ${trdir}/plots
mkdir -p ${trdir}/checkpoints
mkdir -p ${logdir}

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
command="python ${homedir}/kf_train.py --numFolds 5 --fold ${SLURM_ARRAY_TASK_ID} --type hldnn --nodes 130 130 --numEpochs 80 --maxConstits 0"

# Run command
echo "================================"
echo "Will run command ${command}"
$command
echo -e "\nDone!"

# Finally move output files from outfiles to trdir
echo "================================"
echo "Transferring output files..."
mv ${homedir}/outfiles/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.out ${trdir}
mv ${homedir}/outfiles/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.err ${trdir}
