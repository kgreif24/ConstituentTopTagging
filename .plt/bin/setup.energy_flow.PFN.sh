#!/usr/bin/env bash

# Source env
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set name of this model
export CURRENT_MODEL="EFPFN"

# List of all projects
export PATH2PRO="$PATH2HOME/out/AK10UFOSD/rel22p0/energyFlow/PFN"
export LEGENTRY="10 50 80 150 200"
export PROJECTS=$(for num in $LEGENTRY; do echo "$PATH2PRO/efPfn.cufos_${num}.bs_100.ep_100"; done)
export PATH2OUT="$PATH2PLOT/out/energyFlow/PFN"
export TXTONPLT=("#bf{#scale[1.3]{MVAE with neg. Ashman loss}}" "Anti-k_{t} R=0.8" "PFlow jets Pythia + Delphes (ATLAS)" "Trained on 50% QCD and 50% Top")
