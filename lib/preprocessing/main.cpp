// Standard libraries
#include <iostream>
#include <string>
#include <cstdio>
#include <regex>

// ROOT libraries
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TMath.h"

// Other
#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

// Custom
//#include "ppfunc.hpp"


std::vector<std::string> drop_cols (std::vector<std::string> cols)
{
  std::vector<std::string> cols2keep(cols);
  const std::vector<std::string> cols2drop = {"_fjet_", "_clus_"};
  cols2keep.erase(std::remove_if(
    cols2keep.begin(), cols2keep.end(),
    [&cols2drop](const std::string & x) {
      return std::any_of(cols2drop.begin(), cols2drop.end(), [&x](std::string drop){ return x.find(drop) != std::string::npos;} );
    }), cols2keep.end());
  return cols2keep;
}


auto mass = [](const ROOT::RVec<float> & pt, const ROOT::RVec<float> & eta, const ROOT::RVec<float> & phi, const ROOT::RVec<float> & e)
{
  ROOT::RVec<float> m(pt.size());
  for (unsigned int ii = 0; ii<m.size(); ++ii)
    m[ii] = TMath::Sqrt(TMath::Max(0., e[ii]*e[ii] - pt[ii]*pt[ii] * (1. + TMath::SinH(eta[ii])*TMath::SinH(eta[ii]) )));
  return m;
};


auto flip_phi = [](const ROOT::RVec<float> & phi)
{

  ROOT::RVec<float> phi_flip(phi);

  // Find max and min phi of constituents
  float max_phi = ROOT::VecOps::Max(phi);
  float min_phi = ROOT::VecOps::Min(phi);

  // Now if difference between max and min is > pi, we need to flip upper/lower half planes
  if (max_phi - min_phi > TMath::Pi()) {
    // Make vectors with phi+pi and phi-pi
    ROOT::RVec<float> phi_m_pi = phi - TMath::Pi();
    ROOT::RVec<float> phi_p_pi = phi + TMath::Pi();
    // Call Where function to flip upper/lower half plane
    phi_flip = ROOT::VecOps::Where(phi > 0, phi_m_pi, phi_p_pi);
  }

  return phi_flip;
};

auto calc_shift = [](const ROOT::RVec<float> & ang, const ROOT::RVec<float> & w)
{
  float mu = ROOT::VecOps::Dot(ang, w) / ROOT::VecOps::Sum(w);
  return mu;
};

auto apply_eta_shift = [](const ROOT::RVec<float> & eta, const float & mu)
{
  ROOT::RVec<float> eta_shift(eta);
  for (auto & el : eta_shift) el -= mu;
  return eta_shift;
};

auto apply_phi_shift = [](const ROOT::RVec<float> & phi, const float & mu)
{
  ROOT::RVec<float> phi_shift(phi);
  // Loop through phi_shift and apply shift to each element
  for(auto & el : phi_shift)
  {
    el -= mu;
    // Take care of discontinuity in phi, and constrain dphi range to [-pi,pi]
    if (el > TMath::Pi())
      el -= 2*TMath::Pi();
    else if (el < -TMath::Pi())
      el += 2*TMath::Pi();
  }
  return phi_shift;
};

auto sort = [](const ROOT::RVec<float> & x, ROOT::RVec<unsigned long> & pos)
{
  return ROOT::VecOps::Take(x, pos);
};


auto flip = [](const ROOT::RVec<float> & x, const int & par)
{
  return par * x;
};

auto norm = [](const ROOT::RVec<float> & x, ROOT::RVec<float> & w)
{
  ROOT::RVec<float> x_norm(x);
  return x_norm / ROOT::VecOps::Sum(w);
};

auto standardize = [](const ROOT::RVec<float> & x)
{
  float x_mean = ROOT::VecOps::Mean(x);
  float x_stddev = ROOT::VecOps::StdDev(x);
  return (x - x_mean) / x_stddev;
};

auto pca_angle = [](const ROOT::RVec<float> & eta, const ROOT::RVec<float> & phi, const ROOT::RVec<float> & e)
{
  // Assume passed in eta/phi are already centered appropriately

  // Get total energy of this jet basedon the individual constituents (not corrected)
  float e_tot = ROOT::VecOps::Sum(e);

  // Now, all constituents are centered around (0,0) within a range of [-R,R]x[-R,R]
  // 1. Get the means of eta and phi in tansformed system
  float mu_eta = ROOT::VecOps::Dot(eta, e) / e_tot;
  float mu_phi = ROOT::VecOps::Dot(phi, e) / e_tot;
  // 2. Get the second moment of eta and phi needed for variance
  float mu_eta2 = ROOT::VecOps::Dot(eta*eta, e) / e_tot;
  float mu_phi2 = ROOT::VecOps::Dot(phi*phi, e) / e_tot;
  // 3. For correlation matrix
  float mu_eta_phi = ROOT::VecOps::Dot(eta*phi, e) / e_tot;

  // Compute the standard deviations
  float sig_eta2 = mu_eta2 - mu_eta * mu_eta;
  float sig_phi2 = mu_phi2 - mu_phi * mu_phi;
  float sig_eta_phi = mu_eta_phi - mu_eta * mu_phi;

  // Solve the eigenvalue problem and compute the characteristic polynomial
  float lam_neg = 0.5 * (sig_eta2 + sig_phi2 - TMath::Sqrt((sig_eta2 - sig_phi2) * (sig_eta2 - sig_phi2) + 4 * sig_eta_phi * sig_eta_phi));

  // Get direction of first PCA
  float first_pca_eta = sig_eta2 + sig_eta_phi - lam_neg;
  float first_pca_phi = sig_phi2 + sig_eta_phi - lam_neg;

  // The sign ofthe first PCA is ambiguous; let it point in the direction of highest energy
  ROOT::RVec<float> proj = first_pca_eta * eta + first_pca_phi * phi;
  float energy_up = ROOT::VecOps::Sum(ROOT::VecOps::Where(proj>0., e, 0.f));
  float energy_dn = ROOT::VecOps::Sum(ROOT::VecOps::Where(proj<=0, e, 0.f));
  if (energy_dn < energy_up)
  {
    first_pca_eta *= -1;
    first_pca_phi *= -1;
  }

  // Compute the rotation angle by which to rotate the constituents in eta-phi space
  float alpha = TMath::Pi() / 2. + TMath::ATan(first_pca_phi / first_pca_eta);
  // Take care of discontinuity
  if (TMath::Cos(alpha) * first_pca_phi > TMath::Sin(alpha) * first_pca_eta)
    alpha -= TMath::Pi();

  return (-1)*alpha;
};


auto parity = [](const ROOT::RVec<float> & eta, ROOT::RVec<float> e)
{
  float energy_pos = ROOT::VecOps::Sum(ROOT::VecOps::Where(eta>0., e, 0.f));
  float energy_neg = ROOT::VecOps::Sum(ROOT::VecOps::Where(eta<=0., e, 0.f));
  return (energy_pos < energy_neg) ? -1 : 1;
};


auto rot_x = [](const ROOT::RVec<float> & x, ROOT::RVec<float> & y, const float & angle)
{
  ROOT::RVec<float> x_rot(x);
  for (unsigned int ii = 0; ii < x.size(); ++ii)
    x_rot[ii] = x[ii]*TMath::Cos(angle) - y[ii]*TMath::Sin(angle);
  return x_rot;
};


auto rot_y = [](const ROOT::RVec<float> & x, ROOT::RVec<float> & y, const float & angle)
{
  ROOT::RVec<float> y_rot(y);
  for (unsigned int ii = 0; ii < y.size(); ++ii)
    y_rot[ii] = x[ii]*TMath::Sin(angle) + y[ii]*TMath::Cos(angle);
  return y_rot;
};


int main (int argc, char **argv)
{
  // Add commandline parser
  CLI::App app{"App description"};
  std::string fin = "";
  app.add_option("--fin", fin, "Input file");
  std::string fout = "out.root";
  app.add_option("--fout", fout, "Output file");
  std::string treename = "FlatSubstructureJetTree";
  app.add_option("-t,--treename", treename, "Name of the TTree/NTuple");
  CLI11_PARSE(app, argc, argv);

  // Run parallel
  ROOT::EnableImplicitMT();

  // Initialize an RDF
  ROOT::RDataFrame RDF(treename, fin);
  auto RDF2 = RDF.Define("_fjet_sort_idx", "ROOT::VecOps::Argsort((-1) * fjet_clus_pt)")
    // Sort constituents in decreasing pT
    .Define("fjet_sortClus_pt",  sort, {"fjet_clus_pt",  "_fjet_sort_idx"})
    .Define("fjet_sortClus_eta", sort, {"fjet_clus_eta", "_fjet_sort_idx"})
    .Define("fjet_sortClus_phi", sort, {"fjet_clus_phi", "_fjet_sort_idx"})
    .Define("fjet_sortClus_e",   sort, {"fjet_clus_E",   "_fjet_sort_idx"})
    .Define("fjet_sortClus_m",    mass, {"fjet_sortClus_pt", "fjet_sortClus_eta", "fjet_sortClus_phi", "fjet_sortClus_e"})
    // Shift components
    .Define("_fjet_etaShift", calc_shift, {"fjet_sortClus_eta", "fjet_sortClus_e"})
    .Define("_fjet_sortClusCenter_eta", apply_eta_shift, {"fjet_sortClus_eta", "_fjet_etaShift"})
    .Define("_fjet_sortClusFlipped_phi", flip_phi, {"fjet_sortClus_phi"})
    .Define("_fjet_phiShift", calc_shift, {"_fjet_sortClusFlipped_phi", "fjet_sortClus_e"})
    .Define("_fjet_sortClusCenter_phi", apply_phi_shift, {"_fjet_sortClusFlipped_phi", "_fjet_phiShift"})
    // Get rotation angle based on PCA
    .Define("fjet_anglePCA", pca_angle, {"_fjet_sortClusCenter_eta", "_fjet_sortClusCenter_phi", "fjet_sortClus_e"})
    // Shift and rotate
    .Define("_fjet_sortClusCenterRot_eta", rot_x, {"_fjet_sortClusCenter_eta", "_fjet_sortClusCenter_phi", "fjet_anglePCA"})
    .Define("fjet_sortClusCenterRot_phi",  rot_y, {"_fjet_sortClusCenter_eta", "_fjet_sortClusCenter_phi", "fjet_anglePCA"})
    // Deterine the parity of this event
    .Define("fjet_parity", parity, {"_fjet_sortClusCenterRot_eta", "fjet_sortClus_e"})
    // Flip jet based on parity
    .Define("fjet_sortClusCenterRotFlip_eta", flip, {"_fjet_sortClusCenterRot_eta", "fjet_parity"})
    // Normalize scaler components by scaler pT sum
    .Define("fjet_sortClusNormByPt_pt", norm, {"fjet_sortClus_pt", "fjet_sortClus_pt"})
    .Define("fjet_sortClusNormByPt_e",  norm, {"fjet_sortClus_e", "fjet_sortClus_e"});

  // Check if name of input and output file are identical
  if (fin == fout)
  {
    fout.replace(fout.find(".root"), sizeof(".root") - 1, ".tmp.root");
    // Save data
    RDF2.Snapshot(treename, fout, drop_cols(RDF2.GetColumnNames()));
    //Rename file
    std::rename(fout.c_str(), fin.c_str());
  }
  else
    RDF2.Snapshot(treename, fout, drop_cols(RDF2.GetColumnNames()));

  return 0;
}

