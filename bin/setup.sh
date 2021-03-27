#!/usr/bin/env bash
echo "[INFO] Setting up environment from constituent-based DNN tagger"

cat <<'EOF'
   ___  _  ___  __  ______            ______
  / _ \/ |/ / |/ / /_  __/__  ___    /_  __/__ ____ ____ ____ ____
 / // /    /    /   / / / _ \/ _ \    / / / _ `/ _ `/ _ `/ -_) __/
/____/_/|_/_/|_/   /_/  \___/ .__/   /_/  \_,_/\_, /\_, /\__/_/
                           /_/                /___//___/

                                  Christof Sauer (csauer@cern.ch)

EOF

# Some other variables
export CERN_USER=$(whoami)
export USER_INITIAL="$(echo $CERN_USER | head -c 1)"

# Set some path variables
export PATHHOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )"; cd .. >/dev/null 2>&1 && pwd )"
export PATH2BIN=$PATHHOME/bin
export PATH2SRC=$PATHHOME/src
export PATH2DAT=$PATHHOME/dat
export PATH2OUT=$PATHHOME/out
export PATH2ETC=$PATHHOME/etc
export PATH2LIB=$PATHHOME/lib
export PATH2TMP=/tmp/$CERN_USER
export PATH2TOOL=$PATHHOME/tool
export PATH2EOS=/eos/user/c/csauer/data/boostedJetTaggers/constitTopTaggerDNN

# Should we run in debug mode?
if [[ $1 == "-d" ]]; then
  export DEBUG="true"
  set -x
elif [[ $1 == "rebuild" ]]; then
  rm $PATH2BIN/.init
else
  export DEBUG="false"
fi
