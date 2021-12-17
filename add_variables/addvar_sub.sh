#!/bin/bash

# Job sumission script for data generation on gpatlas

#SBATCH --job-name=data_aug
#SBATCH --time=10:00:00 # hh:mm:ss

#SBATCH --partition=atlas

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=5G

#SBATCH --output=./outfiles/Dijet%a.out 
#SBATCH --error=./outfiles/Dijet%a.err

#SBATCH --mail-user=kgreif@uci.edu
#SBATCH --mail-type=ALL

#SBATCH --array=1-12

echo Starting add variables run

listfile="./dat/Dijet${SLURM_ARRAY_TASK_ID}.list"

./add_variables ${listfile}

echo Done!
