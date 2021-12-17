/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
  Copy Pasta: Kevin Greif, 11/2/21
*/

#ifndef jetsubstructureutils_boosttocenterofmass_header
#define jetsubstructureutils_boosttocenterofmass_header

#include <vector>
#include "fastjet/PseudoJet.hh"

std::vector<fastjet::PseudoJet> boost(fastjet::PseudoJet jet, std::vector<fastjet::PseudoJet> constituents);

#endif
