#!/usr/bin/env bash

./slim4pred --fin /eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.426345.Pythia8EvtGen_A14NNPDF23LO_Zprime_tt_flatpT.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228575._000001.tree.root \
          --fout test_result.root \
          --label 1 \
          --selTrain "fjet_m>500000" \
          --selTrue "fjet_m>0"


#  args =  "--branches %s --fin __fin__ --fout __fout__ -t %s --selTrue '__sel__' --selTrain '%s' --label __y__" % (" ".join(info["SlimList"]), info["TreeName"], filter_train)
