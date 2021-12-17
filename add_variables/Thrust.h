#ifndef thrust_header
#define thrust_header

#include <map>
#include "fastjet/PseudoJet.hh"

std::map<std::string, double> thrust(const fastjet::PseudoJet &jet);

#endif

