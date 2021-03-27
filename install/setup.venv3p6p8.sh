#!/usr/bin/env bash

# https://stackoverflow.com/questions/49653354/has-something-changed-recently-with-tensorflow-installation-process

# Get path of this script
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Get project path
PROJECTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; cd .. >/dev/null 2>&1 && pwd )"

# Create directory
mkdir -p $PROJECTPATH/venv

# Create the virtual environment
virtualenv -v -p "/usr/bin/python3.6" $PROJECTPATH/venv/venv3p6

# Activate the virtual environment
. $PROJECTPATH/venv/venv3p6/bin/activate

