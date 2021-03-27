#!/usr/bin/env bash

# https://stackoverflow.com/questions/49653354/has-something-changed-recently-with-tensorflow-installation-process

# Get path of this script
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Get project path
PROJECTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; cd .. >/dev/null 2>&1 && pwd )"

# Source root
. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.18.04/x86_64-centos7-gcc48-opt/bin/thisroot.sh

# Create directory
mkdir -p $PROJECTPATH/venv

# Create the virtual environment
virtualenv -v -p "/usr/bin/python" $PROJECTPATH/venv/venv2p7

# Activate the virtual environment
. $PROJECTPATH/venv/venv2p7/bin/activate

# Update pip
pip install --no-cache-dir --upgrade setuptools

# Install all required packages for this project
pip install --upgrade --force-reinstall --no-cache-dir -r $SCRIPTPATH/requirements.txt

# Install custom packages
for lib in myutils myroot myplt myhep myml; do
  pip install --force-reinstall --no-cache-dir git+https://gitlab.com/csauer/${lib}.git
done

# Run a quick test
python $SCRIPTPATH/test.py
