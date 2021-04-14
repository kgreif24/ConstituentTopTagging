#!/usr/bin/env bash

: '
  Christof Sauer, 2021

  Get prediction of different models.
'

echo "[INFO] Starting script: run.predict.sh"
# Get path of this script
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Source envirmonment
source $SCRIPTPATH/setup2p7.sh

# Output directory to hold files
OUTFILE="$PATH2EOS/predict/${CERN_USER}.prediction4PFN_Phi100-100_F100-50-25-10.ak10ufosd.rel22p0.root"
# Path to trained model
PROJECT="$PATH2OUT/AK10UFOSD/rel22p0/EF/PFN/100/Phi100-100_F100-50-25-10"
# Create output directory
mkdir -p $PATH2EOS/predict

# Start to predict
python $PATHHOME/pred.py --input $PROJECT --fout $OUTFILE
