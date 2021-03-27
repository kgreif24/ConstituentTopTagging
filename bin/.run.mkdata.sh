#!/usr/bin/env bash

: '
  Christof Sauer, 2021

  This script generates a dataset (ROOT + HDF5 file) that can be used to
  train the constituent-based tagger.
'

echo "[INFO] Starting script: run.mkdata.sh"
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
NCONSTIT=200

# Make sure directories exist
mkdir -p  $PATH2EOS/training

# Loop over jet collections
for jc in "${!JCS[@]}"; do
  # Loop over truth label definitions
  for rel in "${!TLS[@]}"; do
    # Path to files for signal and background samples
    SIG="$CONSHOME/dat/${rel}/Top${jc}.list"
    BKG="$CONSHOME/dat/${rel}/Dijet${jc}.list"
#    SIG="$CONSHOME/dat/${rel}/DebugTop.list"
#    BKG="$CONSHOME/dat/${rel}/DebugDijet.list"
    # Build file name of the output file
    FNAME="${CERN_USER}.data4tagger.${JCS[$jc]}.${TLS[$rel]}.pp.nufos_${NCONSTIT}.DELETE"
    # Print command to check if everything is as expected
    python2 $CONSHOME/mkdata.py --truth-label $rel --input-sig $SIG --input-bkg $BKG --n-events 500000 --n-constit $NCONSTIT --fout $PATH2TMP/$FNAME
    # Copy files to EOS directory
#    rsync -rah --progress --remove-source-files $PATH2TMP/$FNAME* $PATH2EOS/training/.
    echo "[INFO] Done"
  done;
done
