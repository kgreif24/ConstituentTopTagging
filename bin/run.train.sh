#!/usr/bin/env bash

: '
  Christof Sauer, 2021

  Train different models.
'

echo "[INFO] Starting script: run.train.sh"
# Get path of this script
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Source envirmonment
source $SCRIPTPATH/setup2p7.sh

ARCH="PFN"
# Check command-line arguments
if [[ $1 != '' ]]; then
  ARCH=$1
fi

# Path to file for training
FILE=$PATH2EOS/training/${CERN_USER}.data4tagger.1M.ak10ufosd.rel22p0.root

BS=500
# Start training of the model
OUTDIR=$PATHHOME/out/AK10UFOSD/rel22p0/EF/$ARCH/$BS/Phi300-300-300_F300-300
$PATHHOME/train.py --input $FILE --outdir $OUTDIR --n-epoch 100 --batch-size $BS --n-train 500000 --Phi-size 300 300 300 --F-sizes 300 300 --architecture $ARCH
