#!/usr/bin/env bash

# Get path of this script
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Setup environment
source $SCRIPTPATH/setup.sh

if [[ $1 == "rebuild" ]]; then
  rm $PATH2BIN/.init2p7
fi

# Check is this is the first time, the setup script has been called
if [ ! -f "$PATH2BIN/.init2p7" ]; then
  echo "[INFO] First call. Setting up environment. This may take a while"
  source $PATHHOME/install/setup.venv2p7.sh
  touch $PATH2BIN/.init2p7
fi

# Source the virtual environment with all required python packages
source $PATHHOME/venv/venv2p7/bin/activate

# Source latest root version
. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.18.04/x86_64-centos7-gcc48-opt/bin/thisroot.sh

# Set PYTHONPATH
export PYTHONPATH=$PATHHOME/venv/venv2p7/lib/python2.7/site-packages:$PYTHONPATH
