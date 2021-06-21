// Standard libraries
#include <iostream>
#include <string>
#include <random>
#include <stdlib.h>
#include <cstdio>
#include <iomanip>
#include <iostream>

// ROOT libraries
#include "ROOT/RDataFrame.hxx"

// Other
#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"


std::string hex_id(const unsigned int & length)
{
  char str[100];
  char hex_characters[]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
  for(unsigned int i=0;i<length;i++)
    str[i]=hex_characters[rand() % 16];
  str[length]=0;
  return std::string(str);
}


int main (int argc, char **argv)
{
  // Add commandline parser
  CLI::App app{"App description"};
  std::string fin = "";
  app.add_option("--fin", fin, "Input file");
  std::string fout = "out.root";
  app.add_option("--fout", fout, "Output file");
  std::string path2tmp = "tmp";
  app.add_option("--path2tmp", path2tmp, "Path where to store the temporary files");
  std::string treename = "FlatSubstructureJetTree";
  app.add_option("-t,--treename", treename, "Name of the TTree/NTuple");
  float split_frac = 0.1;
  app.add_option("-f,--fraction", split_frac, "The fraction of the total number present in each new file.");
  CLI11_PARSE(app, argc, argv);

  // Get a unique identifier
  std::string id = hex_id(20);
  std::cout << "[INFO] The following indentifier will be used: " << id << std::endl;

  // Initialize RDataFrame
  auto logStart = [](ULong64_t e) { if (e == 0) std::cout << "[EVENT LOOP] Started. Get a coffee :)" << std::endl; return true; };
  auto RDF = ROOT::RDataFrame(treename, fin).Filter(logStart, {"rdfentry_"});

  // Get number of events in this file
  unsigned int nEvents = RDF.Count().GetValue();
  // How many files should be created?
  unsigned int nFiles = static_cast<int>(1.0 / split_frac);
  // Number of events in each batch/snapshot
  unsigned int batch_size = nEvents / nFiles;

  // Do not trigger event loop at first snapshot call
  ROOT::RDF::RSnapshotOptions opts;
  opts.fLazy = true;

  using SnapRet_t = ROOT::RDF::RResultPtr<ROOT::RDF::RInterface<ROOT::Detail::RDF::RLoopManager>>; // <- OMG!!!
  std::vector<SnapRet_t> snaps;
  std::vector<std::string> fNames;

  for (unsigned int i = 0; i < nFiles; ++i)
  {
    // Compute range for this batch of data
    unsigned int start = i*batch_size;
    unsigned int stop  = (i+1)*batch_size;
    std::cout << "[INFO] Adding snapshot for the range [" << start << "," << stop << ")" << std::endl;
    // Set file name
    std::stringstream nBatch, bRange;
    nBatch << std::setfill('0') << std::setw(5) << std::to_string(i);
    bRange << std::setfill('0') << std::setw(8) << std::to_string(start)
           << '-' << std::setfill('0') << std::setw(8) << std::to_string(stop);
    fNames.push_back(path2tmp + "/shuffle." +  nBatch.str() + "." + bRange.str() + "." + id + ".root");
    std::cout << "[INFO] Name of this snapshot: " << fNames.back() << std::endl;
    // Add snapshot for this range
    snaps.emplace_back(
      RDF.Range(start, stop).Snapshot(treename, fNames.back(), ".*", opts)
    );
  }

  // Trigger event loop
  *RDF.Count();

  // Shuffle files in vector
  std::cout << "[INFO] Shuffle temporary files" << std::endl;
  auto engine = std::default_random_engine {};
  std::shuffle(std::begin(fNames), std::end(fNames), engine);
  std::cout << "[INFO] Shuffled file list" << std::endl;
  for(auto & el : fNames)
    std::cout << "       |- " << el << std::endl;

  // If no output file is provided, end program
  if (fout == "")
    return 0;
  // Merge files and save result
  std::cout << "[INFO] Merge snapshots" << std::endl;
  auto RDF_merge = ROOT::RDataFrame(treename, fNames).Snapshot(treename, fout);
  std::cout << "[INFO] The following file has been saved: " << fout << std::endl;

  // Debug, stop here to see if tmp files actually exist!
  if (fout == "datagen_out.root") {
    std::cout << "Made it to stop switch, hope for the best! " << std::endl;
    return 0;
  }

  // Delete temporary files
  for (auto & el : fNames)
    if (std::remove(el.c_str()) != 0)
      std::cout << "[ERROR] File `" << el << "` could not be deleted" << std::endl;
    else
      std::cout << "[INFO] File `" << el << "` successfully deleted" << std::endl;

  return 0;
}
