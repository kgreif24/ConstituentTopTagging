#!/usr/bin/env bash

# Set some path varibales
HOME=$(pwd)

# Source root
. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.18.04/x86_64-centos7-gcc48-opt/bin/thisroot.sh
source /afs/cern.ch/work/c/csauer/analyses/boostedJetTaggers/constitTopTaggerDNN/venv/venv2p7/bin/activate

mkdir job
# Run script
OUTDIR="job/predict/AK10UFOSD/rel22p0/EF/PFN/nufos_100.bs_100.ep_100.weighted"
PROJECT="/afs/cern.ch/work/c/csauer/analyses/boostedJetTaggers/constitTopTaggerDNN/out/AK10UFOSD/rel22p0/EF/PFN/nufos_100.bs_100.ep_100.weighted"
#echo $(cat predict.py)
python predict.py --input $PROJECT --outdir $OUTDIR --max-processes 5

tar -zcvf result.nufos_100.bs_100.ep_100.$1.tar.gz job
