import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True


ROOT.gInterpreter.Declare('''
#include <numeric>
#include <vector>
#include <math.h>

// ROOT imports
#include "TLorentzVector.h"
#include "TVector3.h"

typedef ROOT::VecOps::RVec<float> fRVec;

// Namespace for constituent-based operations
namespace reco
{
  TLorentzVector jet(const fRVec pt, const fRVec eta, const fRVec phi, const fRVec e)
  {
    TLorentzVector tJet;
    for (unsigned int i=0; i<pt.size(); ++i)
    {
      TLorentzVector tClus;
      tClus.SetPtEtaPhiE(pt[i], eta[i], phi[i], e[i]);
      tJet += tClus;
    }
    return tJet;
  }

  std::vector<TLorentzVector> jets(const fRVec pt, const fRVec eta, const fRVec phi, const fRVec e)
  {
    std::vector<TLorentzVector> vClus(pt.size());
    for (unsigned int i=0; i<pt.size(); ++i)
    {
      vClus[i].SetPtEtaPhiE(pt[i], eta[i], phi[i], e[i]);
    }
    return vClus;
  }

  std::vector<float> constit_m(const fRVec pt, const fRVec eta, const fRVec phi, const fRVec e)
  {
    std::vector<float> vClus(pt.size());
    for (unsigned int i=0; i<pt.size(); ++i)
    {
      TLorentzVector tJet;
      tJet.SetPtEtaPhiE(pt[i], eta[i], phi[i], e[i]);
      vClus[i] = tJet.M()/1000.;
    }
    return vClus;
  }

  float pt(const fRVec pt, const fRVec eta, const fRVec phi, const fRVec e)
  {
    return jet(pt, eta, phi, e).Pt();
  }

  float eta(const fRVec pt, const fRVec eta, const fRVec phi, const fRVec e)
  {
    return jet(pt, eta, phi, e).Eta();
  }

  float phi(const fRVec pt, const fRVec eta, const fRVec phi, const fRVec e)
  {
    return jet(pt, eta, phi, e).Phi();
  }

  float e(const fRVec pt, const fRVec eta, const fRVec phi, const fRVec e)
  {
    return jet(pt, eta, phi, e).E();
  }

  float m(const fRVec pt, const fRVec eta, const fRVec phi, const fRVec e)
  {
    return jet(pt, eta, phi, e).M();
  }
}

// Namespace for preprocessing
namespace pp
{
  // Temporary structure to sort two vectors simultaneously.
  struct Tuple {
      float first, second;
      Tuple(float _first, float _second) : first(_first), second(_second) {}
  };

  // Sort `vec` based on `ref` in descending order, i.e., the same ordering based on
  // `ref` will be applied to `vec`.
  std::vector<float> sort(const ROOT::VecOps::RVec<float> & vec, const ROOT::VecOps::RVec<float> & ref, const unsigned int nconstit=2, const float pad=0)
  {
    std::vector<Tuple> container;
    for (unsigned int i=0; i<vec.size(); ++i)
    {
      container.push_back(Tuple(vec.at(i), ref.at(i)));
    }
    std::sort(container.begin(), container.end(), [ ]( const Tuple & lhs, const Tuple & rhs ){ return lhs.second > rhs.second;});
    std::vector<float> sorted;
    for (unsigned int i=0; i<container.size(); ++i)
    {
      sorted.push_back(container[i].first);
    }

    // Contrain the number of constituents (entries) in the vectors to `nconstit`. If the
    // number is smaller, the remaining entries will be zero padded.
    if (nconstit!=0)
      sorted.resize(nconstit, pad);
    return sorted;
  }

  // Contrain the number of constituents (entries) in the vectors to `nconstit`. If the
  // number is smaller, the remaining entries will be zero padded.
  std::vector<float> limit_nconstit(const fRVec vec, const unsigned int nconstit=50, const float pad=0)
  {
    std::vector<float> vec_cpy(vec.begin(), vec.end());
    vec_cpy.resize(nconstit, pad);
    return vec_cpy;
  }

  std::vector<float> divby(const fRVec vec, const double denum=1)
  {
    std::vector<float> vecnew(vec.size());
    for (unsigned int i=0; i<vec.size(); ++i)
    {
      vecnew[i] = vec[i] / denum;
    }
    return vecnew;
  }

  namespace norm
  {
    std::vector<float> to_energy(const ROOT::VecOps::RVec<float> & vec, const ROOT::VecOps::RVec<float> & e)
    {
      double energy = ROOT::VecOps::Sum(e);
      std::vector<float> vecnew(vec.size());
      for (unsigned int i=0; i<vec.size(); ++i)
      {
        vecnew[i] = vec[i] / energy;
      }
      return vecnew;
    }
  }

  namespace shift
  {
    // Subtract the total sum of all vector elements from each constituents
    std::vector<float> subtract_pos(const ROOT::VecOps::RVec<float> & vec, const ROOT::VecOps::RVec<float> & weight)
    {
      double energy = ROOT::VecOps::Sum(weight);
      // Get weighted mean
      double pos = 0;
      for (unsigned int i=0; i<vec.size(); ++i)
      {
        pos += vec[i] * weight[i] / energy;
      }

      std::vector<float> subtracted(vec.size(), 0.0);
      for (unsigned int i=0; i<subtracted.size(); ++i)
      {
        if (weight[i]==0) continue;
        subtracted[i] = vec[i] - pos;
      }
      return subtracted;
    }

    // Standardize data to have zero mean and unity vaeiance
    std::vector<float> standardization(const fRVec vec, const double ignore=0)
    {
      // Get number of entries different from `ignore`
      float fN = std::count_if (vec.begin(), vec.end(), [&ignore](float item){return item!=ignore;} );
      if (fN == 0) fN = 1;
      double mu  = mu = ROOT::VecOps::Sum(vec)/fN;
      double var = ROOT::VecOps::Dot(vec, vec)/fN - mu*mu/fN;
      std::vector<float> standardized(vec.size(), 0.0);
      for (unsigned int i=0; i<standardized.size(); ++i)
      {
        standardized[i] = (vec[i] - mu) / (var+1.0e-10);
      }
      return standardized;
    }
  }
}
''')
