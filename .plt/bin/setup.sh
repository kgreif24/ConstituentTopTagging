#!/usr/bin/env bash

bold=$(tput bold)
normal=$(tput sgr0)

# Source environment
source $PATH2HOME/bin/setup.sh

# Get path of this script
export SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PATH2PLOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; cd .. >/dev/null 2>&1 && pwd )"
export PATH2HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; cd ../.. >/dev/null 2>&1 && pwd )"

if [[ "$1" == "EFPFN" ]]; then
  source $SCRIPTPATH/setup.energy_flow.PFN.sh
else
  echo "${bold}[ERROR]${normal} '$1' is not a valid argument."
  exit
fi

echo "${bold}[INFO]${normal} Configuration points to: $CURRENT_MODEL"
