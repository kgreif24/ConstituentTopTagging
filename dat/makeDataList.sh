#!/usr/bin/env bash
find /eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/ -path '*A14NNPDF23LO_jetjet*' -a -name '*.root' -type f > DijetSamples.list
find /eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/ -path '*_Zprime_tt*' -name \*.root -type f > ZprimtTTSamples.list

