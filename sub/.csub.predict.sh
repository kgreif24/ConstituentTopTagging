#!/usr/bin/env sh

DIRNAME="SkimSingleMuonData"
CMSREL="CMSSW_5_3_32"
USER=$(whoami)
mkdir -p tmp

: '
  1. Script to build CMS environment with corresponding release.
     Run the main script `skim_cfg.py` to slim and skim AODs
'

# Write a temporary file that sets up the CMS environment CMS' open data
CMSENV="tmp.cmsenv.sh"
cat << EOF > tmp/$CMSENV
#!/usr/bin/env sh
PATH2DIR=\$(pwd)
######################################
#
# Set up CMS environment and release
#
######################################
. /cvmfs/cms.cern.ch/cmsset_default.sh
# Setup the environment if it does not exist
echo "[INFO] Setting up environment for CMS release $CMSREL"
cmsrel $CMSREL
cd $CMSREL/src
######################################
#
# Move project to environment
#
######################################
git config --global push.default matching
git clone --depth=1 --branch=master https://gitlab.com/csauer/${DIRNAME}.git
cd $DIRNAME
cmsenv
source bin/setup.sh
######################################
#
# Run the program
#
######################################
cmsRun python/skim_cfg.py
######################################
#
# Finalize
#
######################################
mv output/skimmedAOD.root \$PATH2DIR/${USER}.skimmedAOD.root
EOF
chmod +x tmp/$CMSENV


: '
  2. Run singularity with CMS image of scientoific linux 6
'

RUNEXE="tmp.run.skimAOD.sh"
cat << EOF > tmp/$RUNEXE
#!/usr/bin/env sh
######################################
#
# Submit job(s) to computer cluster
#
######################################
printf "Start time: "; /bin/date
printf "Job is running on node: "; /bin/hostname
printf "Job running as user: "; /usr/bin/id
printf "Job is running in directory: "; /bin/pwd
. /cvmfs/cms.cern.ch/cmsset_default.sh
cmssw-cc6 --bind /cvmfs --bind \$(pwd) --command-to-run sh $CMSENV
echo "[INFO] Done!"
EOF
chmod +x tmp/$RUNEXE


: '
  3. Submission script for comupter cluster
'

SUBEXE="tmp.sub.skimAOD.sub"
cat << EOF > tmp/$SUBEXE
Universe                = vanilla
executable              = tmp/$RUNEXE
arguments               = \$(ClusterId)\$(ProcId)
output                  = out/skimAOD.\$(ClusterId).\$(ProcId).out
error                   = err/skimAOD.\$(ClusterId).\$(ProcId).err
log                     = log/skimAOD.\$(ClusterId).log
transfer_output_files   = out/${USER}.skimmedAOD.root
when_to_transfer_output = ON_Exit
transfer_input_files    = tmp/$CMSENV, tmp/$RUNEXE
queue
EOF


: '
  4. Submit to cluster
'

# Submit to cluster
condor_submit tmp/$SUBEXE
