#!/bin/bash

# Job sumission script for data generation on gpatlas

#SBATCH --job-name=data_aug1
#SBATCH --time=10:00:00 # hh:mm:ss

#SBATCH --partition=atlas

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=5G

#SBATCH --output=./outfiles/Zprime%a.out 
#SBATCH --error=./outfiles/Zprime%a.err

#SBATCH --mail-user=kgreif@uci.edu
#SBATCH --mail-type=ALL

#SBATCH --array=1-3

echo Starting add variables run

listfile="./dat/Zprime${SLURM_ARRAY_TASK_ID}.list"

echo Using file ${listfile}

./add_variables ${listfile}

echo Done!
