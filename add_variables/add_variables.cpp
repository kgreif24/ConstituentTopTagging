/* add_variables.cpp - This script will define an object that 
calculated extra jet substructure variables using fastjet. In
particular we want to add variables to root files that are 
missing from hl tagger implementation.

Author: Kevin Greif
c++17
Last updated 11/3/21
*/

#include <iostream>
#include <fstream>

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TTreeReader.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/contrib/EnergyCorrelator.hh"
#include "fastjet/contrib/Nsubjettiness.hh"
#include "fastjet/contrib/Njettiness.hh"
#include "fastjet/contrib/NjettinessPlugin.hh"
#include "fastjet/ClusterSequence.hh"

#include "Thrust.h"

using std::cout, std::endl;


// Define a class to augment jets. We'll use the TTreeReader
// class to loop over a root file, and extract the information
// we need to calculate substructure variables using fastjet.
class JetAugment {

public: 

  // Vector for holding jet constituents
  std::vector<fastjet::PseudoJet> towers;

  // Pseudo jet for holding reclustered jet, along with cluster sequence
  fastjet::ClusterSequence ReclusterConsts;
  fastjet::PseudoJet jet;

  // Process, calculate, and build functions
  int process(std::string read_filename, std::string write_filename);
  void build();
  float L2();
  float L3();
  float Tau4wta();
  float ThrustMajor();
  void calculate();

};

int JetAugment::process(std::string read_filename, std::string write_filename) {

  // First open up read file
  // Convert filename to char array
  char* read_char_name;
  read_char_name = &read_filename[0];
  // Open the file
  TFile *read = new TFile(read_char_name, "UPDATE");

  // Set up a TTreeReader (for reading) and  TTree (for writing)
  TTreeReader tree_reader("FlatSubstructureJetTree", read);
  TTree *tree = (TTree*) read->Get("FlatSubstructureJetTree");

  // Set up reader values to recieve data from TTreeReader
  TTreeReaderValue<int> num_constits(tree_reader, "fjet_numConstituents");
  TTreeReaderValue<std::vector<float>> cons_pt(tree_reader, "fjet_clus_pt");
  TTreeReaderValue<std::vector<float>> cons_eta(tree_reader, "fjet_clus_eta");
  TTreeReaderValue<std::vector<float>> cons_phi(tree_reader, "fjet_clus_phi");
  TTreeReaderValue<std::vector<float>> cons_e(tree_reader, "fjet_clus_E");

  // Set up new branches to accept calculated values
  float L2val, L3val, tau4val, thrustval;
  TBranch *bL2 = tree->Branch("fjet_L2", &L2val, "fjet_L2/F");
  TBranch *bL3 = tree->Branch("fjet_L3", &L3val, "fjet_L3/F");
  TBranch *tau4wta = tree->Branch("fjet_Tau4_wta", &tau4val, "fjet_Tau4_wta/F");
  TBranch *thrustMaj = tree->Branch("fjet_ThrustMaj", &thrustval, "fjet_ThrustMaj/F");
  tree->SetBranchAddress("fjet_L2", &L2val);
  tree->SetBranchAddress("fjet_L3", &L3val);
  tree->SetBranchAddress("fjet_Tau4_wta", &tau4val);
  tree->SetBranchAddress("fjet_ThrustMaj", &thrustval);

  // Find number of jets in tree
  int num_jets = tree_reader.GetEntries();

  // Now loop through tree
  while (tree_reader.Next()) {

    // First clear constituents vector
    towers.clear();

    // And loop through constituents to add new ones
    for (int i = 0; i < *num_constits; i++) {

      fastjet::PseudoJet p;
      p.reset_momentum_PtYPhiM((*cons_pt)[i], (*cons_eta)[i], (*cons_phi)[i], (*cons_e)[i]);
      towers.push_back(p);

    }

    // We now have vector of pseudo jets for this event. Recluster to build jet
    build();

    // We now calculate variables
    L2val = L2();
    L3val = L3();
    tau4val = Tau4wta();
    thrustval = ThrustMajor();

    // Now fill branches
    bL2->Fill();
    bL3->Fill();
    tau4wta->Fill();
    thrustMaj->Fill();

  }

  // Create write file
  char* write_char_name;
  write_char_name = &write_filename[0];
  TFile *write = new TFile(write_char_name, "RECREATE");
  
  // Print and write the tree
  // tree->Print();
  // write->WriteObject(tree, "FlatSubstructureJetTree");
  write->cd();
  tree->CloneTree()->Write();
  
  // Close files
  read->Close();
  write->Close();

  return num_jets;

}

void JetAugment::build() {

  // Recluster jets
  fastjet::JetDefinition jet_def(fastjet::JetDefinition(fastjet::antikt_algorithm, 1.2));
  ReclusterConsts = fastjet::ClusterSequence(towers, jet_def);
  std::vector<fastjet::PseudoJet> InclusiveJets = sorted_by_pt(ReclusterConsts.inclusive_jets());

  // Get leading jet (should be the only one) together with it's constituents. Set this to jet variable
  jet = InclusiveJets[0];

}

float JetAugment::L2() {

  // Calculate energy correlation functions to get L2
  fastjet::contrib::EnergyCorrelatorGeneralized ECFG_3_3_1(3, 3, 1, fastjet::contrib::EnergyCorrelator::pt_R);
  fastjet::contrib::EnergyCorrelatorGeneralized ECFG_2_1_2(1, 2, 2, fastjet::contrib::EnergyCorrelator::pt_R);

  double ecfg_3_3_1 = ECFG_3_3_1.result(jet);
  double ecfg_2_1_2 = ECFG_2_1_2.result(jet);

  // Calculate L2
  float L2 = -999;
  if (ecfg_2_1_2 > 1e-8) {
    L2 = ecfg_3_3_1 / pow(ecfg_2_1_2, (3.0/2.0));
  }

  return L2;

}

float JetAugment::L3() {

  // Calculate energy correlation functions to get L3
  fastjet::contrib::EnergyCorrelatorGeneralized ECFG_3_3_1(3, 3, 1, fastjet::contrib::EnergyCorrelator::pt_R);
  fastjet::contrib::EnergyCorrelatorGeneralized ECFG_3_1_1(1, 3, 1, fastjet::contrib::EnergyCorrelator::pt_R);

  double ecfg_3_3_1 = ECFG_3_3_1.result(jet);
  double ecfg_3_1_1 = ECFG_3_1_1.result(jet);

  // Calculate L3
  float L3 = -999;
  if (ecfg_3_3_1 > 1e-8) {
    L3 = ecfg_3_1_1 / pow(ecfg_3_3_1, (1.0/3.0));
  }

  return L3;

}

float JetAugment::Tau4wta() {

  // Setup calculation and calculate, using cutoff measure at 1TeV
  fastjet::contrib::WTA_KT_Axes wta_kt_axes;
  fastjet::contrib::NormalizedCutoffMeasure measure(1.0, 1.0, 1000000);
  fastjet::contrib::Nsubjettiness Tau4WTA(4, wta_kt_axes, measure);

  float tau4 = -999;
  tau4 = Tau4WTA.result(jet);

  return tau4;

}

float JetAugment::ThrustMajor() {

  // Just need to passs jet into thrust function
  std::map<std::string, double> res_t;
  res_t["ThrustMaj"] = -999;
  res_t["ThrustMin"] = -999;
  res_t = thrust(jet);
  
  return (float) res_t["ThrustMaj"];
  
}


// This function just helps with naming files in the main program
std::string ReplaceString(std::string subject, const std::string& search,
                          const std::string& replace) {
    size_t pos = 0;
    while((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
    return subject;
}


int main(int argc, char* argv[]) {

  // We want to loop through the list of files in .list files in /dat folder
  cout << "Using input file " << argv[1] << endl;

  // Make fstream object and open up the .list file
  std::fstream listfile;
  listfile.open(argv[1], std::ios::in);

  // If we have a good open, we want to loop through the lines of this file
  if (listfile.is_open()) {

    // Here loop through lines using getline function
    std::string infile;
    while (getline(listfile, infile)) {

      // Now we want to build the name of our write files
      std::string outfile = ReplaceString(infile, "ConstituentTaggers", "ModTaggingData");
      cout << "Will create file: " << outfile << endl;
      
      // Finally, make jet augment object and run the processing
      JetAugment aug;
      int file_jets = aug.process(infile, outfile);
      cout << "Number of jets in file: " << file_jets << endl;
      cout << endl;

    }

  }

}
