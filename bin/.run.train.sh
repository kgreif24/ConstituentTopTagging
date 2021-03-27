#!/usr/bin/env bash

: '
  Christof Sauer, 2021

  Train different models.
'

echo "[INFO] Starting script: run.train.sh"
# Get path of this script
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Source envirmonment
source $SCRIPTPATH/setup.sh
[ "$DEBUG" == "true" ] && set -x

# Associative array of jet collections and its name
declare -A JCS
JCS["AntiKt10UFOCSSKSoftDropBeta100Zcut10Jets"]="AK10UFOSD"
# A list of truth labels
declare -A TLS
TLS["rel22p0"]="rel22p0"
# Number of constituents to keep and number of events (train+valid)
NCONSTIT="100"
ARCH=PFN

for n_ufos in $NCONSTIT; do
  for jc in "${!JCS[@]}"; do
    # Loop over truth label definitions
    for rel in "${!TLS[@]}"; do
      # Set ouput directory
      OUTDIR=$CONSHOME/out/${JCS[$jc]}/${TLS[$rel]}/EF/$ARCH/nufos_${n_ufos}.bs_100.ep_100
      mkdir -p $OUTDIR
      # Path to file for training
      FILE=$PATH2EOS/training/csauer.data4tagger.AK10UFOSD.rel22p0.pp.nufos_200.root
      $CONSHOME/train.py --input $FILE --outdir $OUTDIR --n-epoch 10 --batch-size 100 --n-constit $n_ufos --format "sequential" --architecture $ARCH
    done
  done
done
