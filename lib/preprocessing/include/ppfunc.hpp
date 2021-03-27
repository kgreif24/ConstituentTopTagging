#ifndef PPFUNC
#define PPFUNC


// Standard libraries
#include <iostream>
#include <string>
#include <cstdio>
#include <algorithm>

// ROOT libraries
#include "ROOT/RDataFrame.hxx"
#include "TH2D.h"
#include "TFile.h"
#include "TMath.h"


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


auto shift_eta = [](const ROOT::RVec<float> & eta, const ROOT::RVec<float> & w)
{
  ROOT::RVec<float> eta_shift(eta);
  float mu = ROOT::VecOps::Dot(eta, w) / ROOT::VecOps::Sum(w);
  for(auto & el : eta_shift) el -= mu;
  return eta_shift;
};


auto shift_phi = [](const ROOT::RVec<float> & phi, const ROOT::RVec<float> & w)
{
  ROOT::RVec<float> phi_shift(phi);
  float mu = ROOT::VecOps::Dot(phi, w) / ROOT::VecOps::Sum(w);
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


auto pca_angle = [](const ROOT::RVec<float> & eta, const ROOT::RVec<float> & phi, const ROOT::RVec<float> & e)
{
  // Shift eta and phi of the constituents accordingly
  ROOT::RVec<float> deta = shift_eta(eta, e);
  ROOT::RVec<float> dphi = shift_phi(phi, e);

  // Get total energy of this jet basedon the individual constituents (not corrected)
  float e_tot = ROOT::VecOps::Sum(e);

  // Now, all constituents are centered around (0,0) within a range of [-R,R]x[-R,R]
  // 1. Get the means of eta and phi in tansformed system
  float mu_eta = ROOT::VecOps::Dot(deta, e) / e_tot;
  float mu_phi = ROOT::VecOps::Dot(dphi, e) / e_tot;
  // 2. Get the second moment of eta and phi needed for variance
  float mu_eta2 = ROOT::VecOps::Dot(deta*deta, e) / e_tot;
  float mu_phi2 = ROOT::VecOps::Dot(dphi*dphi, e) / e_tot;
  // 3. For correlation matrix
  float mu_eta_phi = ROOT::VecOps::Dot(deta*dphi, e) / e_tot;

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
  ROOT::RVec<float> proj = first_pca_eta * deta + first_pca_phi * dphi;
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

#endif // PPFUNC_INCLUDE
