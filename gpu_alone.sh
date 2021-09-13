#!/bin/bash

# This is a script for running jobs on a gpu box. Meant to be used
# when there is no scheduler, so no funny buisness!

# To start, we need to decide the name of our model design folder.
# Set this as a keyword argument
modelname=$1

##################### Job Loop ###################

# Loop through the number of runs we want to do
run=0
while [ $run -lt $2 ]
do

    # Next we need to decide which run of the model design this is. We just
    # take the first available integer starting from one.
    counter=1
    while :
    do
	if [ ! -d training/${modelname}/run_$counter ]
	then
	    runname=$counter
	    break
	fi
	((counter++))
    done

    # Finally we need to figure out which fold of the training data we
    # will reserve for validation. This will just be the run number modulo
    # the number of folds (set below)
    numfolds=5
    fold=`expr $runname % $numfolds`

    # Now that we have all of our run information, make directory tree
    homedir=$(pwd)
    trdir="${homedir}/training/${modelname}/run_${counter}"
    logdir="${homedir}/training/${modelname}/logs"
    mkdir -p ${trdir}/plots
    mkdir -p ${trdir}/checkpoints
    mkdir -p ${logdir}

    # On to running the job! Redirect output from here on to log files in trdir

    {
	
	# Start by printing out hostname and date of job
	echo "Hello, here's some info: "
	hostname; date
	echo "================================"

	# We want output files to sit in trdir, so make that working directory
	cd $trdir
	echo "In directory ${trdir}"
	ls -lrth

	# Next build command to run python training script
	command="python ${homedir}/kf_train.py --numFolds ${numfolds} --fold ${fold} --type efn --phisizes 80 80 --fsizes 80 50 25 10 --numEpochs 50 --maxConstits 80"
	# command="python ${homedir}/pr_train.py --type efn --phisizes 80 80 --fsizes 80 50 25 10 --numEpochs 100"
	# command="python ${homedir}/up_train.py --numFolds ${numfolds} --fold ${fold} --type efn --phisizes 80 80 --fsizes 80 50 25 10 --numEpochs 50 --maxConstits 80"

	# Run command
	echo "================================"
	echo "Will run command ${command}"
	# $command
	echo -e "\nDone!"

    } > ${trdir}/output.log 2>${trdir}/error.log

    # Finish up job
    echo "Done with run ${run}"
    cd $homedir
    ((run++))

done


