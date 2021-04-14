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

BS=200
NC=3
for nconstit in 3
do
  # Start training of the model
  OUTDIR=$PATHHOME/out/AK10UFOSD/rel22p0/EF/$ARCH/bs${BS}_nc${nconstit}/Phi100-100_F100-50-25-10/
  $PATHHOME/train.py --input $FILE --outdir $OUTDIR --n-epoch 50 --batch-size $BS --n-train 300000 --n-constit ${nconstit} --Phi-size 100 100 --F-sizes 50 25 10 --architecture $ARCH &
done
