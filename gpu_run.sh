#!/bin/bash

# This is a script for submitting training jobs on the HPC3 cluster.
# It will use GPU accleration!

# Set up sbatch arguments

#SBATCH --job-name=PNETs2                         ## Name of the job.
#SBATCH -A kgreif                                 ## account to charge 
#SBATCH -p free-gpu                               ## partition/queue name
#SBATCH --gres=gpu:V100:1                         ## Use only 1 GPU
#SBATCH --nodes=1                                 ## (-N) number of nodes to use
#SBATCH --tmp=80G                                 ## Request 80GB of scratch to hold data

#SBATCH --array=1-5

#SBATCH --error=./outfiles/%x_%a.err       ## error log file
#SBATCH --output=./outfiles/%x_%a.out      ## output file

#SBATCH --mail-type=END,FAIL                      ## Send email
#SBATCH --mail-user=kgreif@uci.edu                ## to this address

# Set tmpdir to correct location
export TMPDIR=/tmp/tt_data
trainname="$TMPDIR/train_${SLURM_ARRAY_TASK_ID}.h5"
mkdir -p $TMPDIR 

# Print out hostname and date of job
echo "Found a node, here's some info: "
hostname; date
echo $SLURM_JOB_NAME
echo $SLURM_JOB_ID
echo $SLURM_ARRAY_TASK_ID
echo "================================"

# Transfer file from disk to scratch
echo "Now transferring file from /pub to /tmp"
cp /pub/kgreif/samples/h5dat/train_s2_ln.h5 ${trainname}
ls $TMPDIR 

# Set up directory tree
homedir="/data/homezvol0/kgreif/toptag/ConstituentTopTagging"
trdir="${homedir}/training/${SLURM_JOB_NAME}/run_${SLURM_ARRAY_TASK_ID}"
logdir="${homedir}/training/${SLURM_JOB_NAME}/logs"
rm -rf ${trdir}
mkdir -p ${trdir}/plots
mkdir -p ${trdir}/checkpoints
mkdir -p ${logdir}

# We want output files to sit in trdir, so make that working directory
cd $trdir
echo "In directory ${trdir}"
ls -lrth

# Next build command to run python training script
command="python ${homedir}/kf_train.py --numFolds 10 --fold ${SLURM_ARRAY_TASK_ID} --type pnet --n_blocks 3 --convs 64 224 384 --block_depth 3 3 3 --pooling max --nodes 125 --dropout 0.1 -lr 4.2e-4 -b 256 -N 50 --file ${trainname} --dir ${trdir} --logdir ${logdir}"

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

# Delete tmp files
rm -rf $TMPDIR
