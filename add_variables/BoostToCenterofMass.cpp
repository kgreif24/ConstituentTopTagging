/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
  Copy Pasta: Kevin Greif, 11/2/21
*/

#include "BoostToCenterofMass.h"
#include "TLorentzVector.h"

std::vector<fastjet::PseudoJet> boost(fastjet::PseudoJet jet, std::vector<fastjet::PseudoJet> constit_pseudojets) {
  std::vector<fastjet::PseudoJet> clusters;
  if(jet.e() < 1e-20) { // FPE
    return clusters;
  }

  double bx = jet.px()/jet.e();
  double by = jet.py()/jet.e();
  double bz = jet.pz()/jet.e();

  if(bx*bx + by*by + bz*bz >= 1) { // Faster than light
    return clusters;
  }

  for(unsigned int i1=0; i1 < constit_pseudojets.size(); i1++) {
    TLorentzVector v;
    v.SetPxPyPzE(constit_pseudojets.at(i1).px(), constit_pseudojets.at(i1).py(),constit_pseudojets.at(i1).pz(),constit_pseudojets.at(i1).e());
    v.Boost(-bx,-by,-bz);
    fastjet::PseudoJet v2(v.Px(), v.Py(), v.Pz(), v.E());
    clusters.push_back(v2);
  }

  return clusters;
}
