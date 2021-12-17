#!/usr/bin/env bash
find /DFS-L/DATA/whiteson/kgreif/ModTaggingData/ -path '*A14NNPDF23LO_jetjet*' -a -name '*.root' -type f > DijetSamples.list
find /DFS-L/DATA/whiteson/kgreif/ModTaggingData/ -path '*_Zprime_tt*' -name \*.root -type f > ZprimeTTSamples.list

