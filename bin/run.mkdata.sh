#!/usr/bin/env bash

: '
  Christof Sauer, 2021

  This script generates a dataset (ROOT) that can be used to
  train the constituent-based DNN tagger
'

echo "[INFO] Starting script: run.mkdata.sh"
# Get path of this script
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $SCRIPTPATH/setup2p7.sh

# Make sure directories exist
mkdir -p $PATH2EOS/training

SIG="$PATHHOME/dat/rel22p0/TopAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list"
BKG="$PATHHOME/dat/rel22p0/DijetAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list"
#SIG="$PATHHOME/dat/rel22p0/DebugTop.list"
#BKG="$PATHHOME/dat/rel22p0/DebugDijet.list"

FNAME="$PATHHOME/tmp/training/${CERN_USER}.data4tagger.1M.ak10ufosd.rel22p0.root"
$PATHHOME/mkdata.py --truth-label rel22p0 --input-sig $SIG --input-bkg $BKG --n-events 1000000 --fout $FNAME

# Copy files to EOS directory
rsync -rah --progress --remove-source-files $FNAME $PATH2EOS/training/.
echo "[INFO] Done"
