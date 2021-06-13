#!/usr/bin/env bash
find /eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/ -path '*A14NNPDF23LO_jetjet*' -a -name '*.root' -type f > DijetAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list
find /eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/ -path '*_Zprime_tt*' -name \*.root -type f > TopAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list

