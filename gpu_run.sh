#!/bin/bash

# This is a script for submitting training jobs on the HPC3 cluster.
# It will use GPU accleration!

# Set up sbatch arguments

#SBATCH --job-name=PFNs1                         ## Name of the job.
#SBATCH -A kgreif                                 ## account to charge 
#SBATCH -p free-gpu                               ## partition/queue name
#SBATCH --gres=gpu:V100:1                         ## Use only 1 GPU
#SBATCH --nodes=1                                 ## (-N) number of nodes to use
#SBATCH --tmp=50G                                 ## Request 80GB of scratch to hold data

#SBATCH --array=1-10

#SBATCH --error=./outfiles/%x_%a.err       ## error log file
#SBATCH --output=./outfiles/%x_%a.out      ## output file

#SBATCH --mail-type=END,FAIL                      ## Send email
#SBATCH --mail-user=kgreif@uci.edu                ## to this address

# For puposes of file copying, we want jobs to start running in 5 second increments
waitcount=0
while [ $waitcount -le ${SLURM_ARRAY_TASK_ID} ]
do
	sleep 5
	waitcount=$(($waitcount + 1))	
done

# Setup environment
echo "Building software environment"
module load cuda/11.7.1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH

# Print out hostname and date of job
echo "Found a node, here's some info: "
hostname; date
echo $SLURM_JOB_NAME
echo $SLURM_JOB_ID
echo $SLURM_ARRAY_TASK_ID
echo "================================"

# Set tmpdir to correct location
export TMPDIR=/tmp/tt_data
tempname="$TMPDIR/temp.h5"
trainname="$TMPDIR/train.h5"

# If temp data file does not exist, transfer file from disk to scratch
if [ ! -f $tempname ]
then
	echo "Now transferring file from /pub to /tmp"
	mkdir -p $TMPDIR
	cp /pub/kgreif/samples/h5dat/train_s2_ln_small.h5 $tempname
	mv $TMPDIR/temp.h5 $trainname
	ls $TMPDIR
# If temp data does exist, wait until transfer finishes and then continue
else
	while [ ! -f $trainname ]; do sleep 30; done
fi

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
command="python ${homedir}/kf_train.py --numFolds 10 --fold ${SLURM_ARRAY_TASK_ID} --type pfn --phisizes 250 250 250 --fsizes 500 500 500 --latent_dropout 0.084 --dropout 0.036 -lr 7.9e-5 -b 256 --numEpochs 75 --maxConstits 80 --file ${trainname} --dir ${trdir} --logdir ${logdir}"

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

# Delete mp files
rm -rf $TMPDIR
