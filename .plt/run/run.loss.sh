#!/usr/bin/env bash

# Get project path
PROJECTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; cd .. >/dev/null 2>&1 && pwd )"

# Setup environment
source $PROJECTPATH/bin/setup.sh $1

if [ -n "$PROJECTS" ]; then
  FILES=$(for path in $PROJECTS; do echo "$path/data/history.json"; done)
  $PATH2HOME/plt/loss.py --inputs $FILES --entries $LEGENTRY --outdir $PATH2OUT --text "${TXTONPLT[@]}"
fi

