# Constituent-based (DNN) Top-Tagger (ConstitTopTagger)

This repository is part of my (Christof Sauer) service task at CERN 2019/20 in the ATLAS *Jet-Tagging and Energy Scale Factor* group <3.

## Description of the task

# Author(s)

If something is unclear or you encounter problems or bugs of any kind, do not hesitate to contact:

- Christof Sauer <christof.sauer@cern.ch> Ruprecht-Karls Universität Heidelberg, Germany (main developer)

# 1 Set up working environment and installation

In order to run the project on, e.g., LXplus, a virtual environment must be installed to install python packages required by this project (in particular machine learning libraries as well as some of my own/personal packages that encapsulate and provide most of the functionalities). The configured shell script *setup_venv2.sh* and *setup_venv3.sh*, which are located in the *install* directory, can be used to set up the virtual environment for python2 and python3 respectively. Furthermore, it installs all necessary packages listed in the file *requirements.txt*. To set up the entire environment, just execute the following command in the main directory

```bash
make
```

which will execute the Makefile and install all necessary software. Please be patient; the installation will take some time to be finished. However, afterwards we're good to go. This script creates an environment directory, *venv* with *venv/venv2* and *venv/venv3*, which are located in the root directory of this repository. The next time you start a new session on LXplus, you have to activate/source this virual environment. To do so, just execute the following script after you log in into a new session:

```bash
source venv/bin/activate
```

If you want to delete all python packages that have been installed, just remove the directory of the virtual environment and all traces vanish in the digital orcus. Alternatively, you may also directly source the shell script *bin/setup.sh* in the binary folder in order to activate the virtual environment and to source the latest version of ROOT (v6.18.04) all at once.

```bash
source bin/setup.sh
```

This last command you have to execute all the time you connect to LXPLUS. That's all! We're ready to start with the actual project.

## Note
By executing *requirements.txt*, the following (light) repositories will be installed

```bash
https://gitlab.com/csauer/myutils.git
https://gitlab.com/csauer/myroot.git
https://gitlab.com/csauer/myplt.git
https://gitlab.com/csauer/myhep.git
https://gitlab.com/csauer/myml.git
```

Those repositories are currently **not** hosted by CERN's GitLab; however, all projects listed above are public and so is the source code. By calling *make*, all repositories are automatically cloned into the *lib* directory such that the code may be modified according to your needs. However, it's recommended to utilize pip for the installation (which is also done when calling *make*); otherwise, the smooth functionality of the project is not guaranteed.

### In a nutshell:

To set up the environemnt execute the Makefile in home

```bash
make
```

All the time to connect to LXPLUS, source the script

```bash
source bin/setup.sh
```

to activate the virtual environment and to set up ROOT.

# For the impatient user

In case you just want to produce some results, execute the following commands in teh given order. Be aware that this will take more than a day to finish :).

## 1. Generate training data

To generate a data set for training and quick evaluation

```bash
./bin/run.mkdata.sh
```

This will create a file **PATH2EOS/training/${CERN_USER}.data4tagger.1M.ak10ufosd.rel22p0.root** on your EOS workspace. Furthermore, the script automatically generates some control plots that can be found in the directory *out/data*. The histograms are stores in the ROOT files while the corresponding PDFs are located in the *out/data/plt* folder.

## 2. Train the model

To train a PFN (cf. https://arxiv.org/abs/1810.05165) execute the following command

```bash
./bin/run.train.sh
```

This will create a project directory *out/AK10UFOSD/rel22p0/EF/ARCHITECTURE/BATCH_SIZE/Phi300-300-300_F300-300* which contains some metadata, the model configutation as well as some training metrics (loss and accuracy).

After training, a quick evaluation will be performed and the output will be stored in the aforementioned directory (all reults related to this evaluation have the suffix "qick" attached). This will be very close to the final performance by only takes a fraction of time!

## 3. Get prediction

Now, we evaluate the entire data set by getting the classification score of the tagger.

```bash
./bin/run.pred.sh
```

This will create a file **PATH2EOS/predict/CERN_USER.prediction4PFN.ak10ufosd.rel22p0.root** on your EOS work space.

Comment: This step is very time consuming. Depending on your system, it will run for roughly 10-15 hours. I'm still trying to further optimize this step by utilising a multi threading approach.

## 4. Evaluate the tagger

The last step is to evaluate the tagger performance at a given working point

```bash
./bin/run.eval.sh
```

The output ROOT files and plots are automatically added to the project directory from step 2 such that all infomation is conjtained in one folder.

Comment: Currently, the full data set and the training set are not othorgonal, i.e., events that have been used to train the tagger are also used to evaluate its performance. This will change in the future. However, provided that the DNN did not overfit the training data (check loss and accuracy), this won't result in a bias. Besides that, the training data only is about 1% of the total data set. Hence, the effect is small.


# Step-by-Step instruction

This section gives an example of the full analysis cycle from the generation of the data to train the DNN model to its evaluation to obtain the background rejection plots shown above. This is, of course, only an example and many things like paths etc may – and definitely! – will change in the future. However, all steps described in this section should be rather easy to modify according to your needs. If nothing helps or you get stuck, please do not hesitate to get in touch with me; I'd be pleased to help.


## 1. Training data

The first step is to prepare the so-called *list* files that will be located in the *data* directory. Those files do not contain any data, but only the location of your files. The following snippet of the file *data/rel20.7/DijetAntiKt10LCTopoTrimmedPtFrac5SmallR20Jets.list* gives an example for dijet (main background to hadronically decaying ttbar pairs) ntuples located under EOS:

**dat/rel22p0/DijetAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list**
```bash
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228583._000001.tree.root
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228583._000002.tree.root
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228583._000003.tree.root
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228583._000004.tree.root
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.364703.Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3WithSW.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228583._000005.tree.root
...
```

**dat/rel22p0/TopAntiKt10UFOCSSKSoftDropBeta100Zcut10Jets.list**
```bash
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.426345.Pythia8EvtGen_A14NNPDF23LO_Zprime_tt_flatpT.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228575._000001.tree.root
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.426345.Pythia8EvtGen_A14NNPDF23LO_Zprime_tt_flatpT.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228575._000002.tree.root
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.426345.Pythia8EvtGen_A14NNPDF23LO_Zprime_tt_flatpT.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228575._000003.tree.root
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.426345.Pythia8EvtGen_A14NNPDF23LO_Zprime_tt_flatpT.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228575._000004.tree.root
/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysisRel21/FlattenerOutput/ConstituentTaggers/user.csauer.426345.Pythia8EvtGen_A14NNPDF23LO_Zprime_tt_flatpT.Flattener_v2.UFOVSD.Constit_tree.root/user.csauer.23228575._000005.tree.root
...
```

By the time you're reading this file, those files might have been deleted already; furthermore, you probably wouldn't be able to read them anyway due to missing reading permission. So it's time to make your own files. By the way, the *.list* files in the *dat* directory have been generated by the respective shell scripts such as *AntiKt10UFOCSSKSoftDropBeta100Zcut10.sh*; just adapt them to ýour needs or copy paste each path individually.

Once the files are prepared, generating a training set is a piece of cake: all that os left to do is to call the python script *mkdata.py* with your specific arguments. To get a list of all arguments just type (do not forget to source the virtual environment by calling `source bin/setup.sh to avoid import errors):

```python
python mkdata.py -h
```

This will result in

```bash
usage: mkdata.py [-h] [--fout FOUT] --n-events N_EVENTS
                 [--truth-label TRUTH_LABEL] --input-bkg INPUT_BKG --input-sig INPUT_SIG
```

The generated files are then given by:

```bash
PATH2EOS/training/${CERN_USER}.data4tagger.1M.ak10ufosd.rel22p0.root
```

The script automatically generates some control plots that can be found in the directory *out/data*. The histograms are stores in the ROOT files while the corresponding PDFs are located in the *out/data/plt* folder.

### Comment: trimming

In most cases, the underlying ntuples contain an abundance of variables that are not of interest in this study. To define the list of branches that you want to keep in your generated data set, you have to edit the configuration file *trim_slim.ini* located in *etc*. One example of such a selection of variables would be for instance:

```bash
[slim]
branches=fjet_m;fjet_pt;fjet_eta;fjet_phi;fjet_truthJet_pt;fjet_truthJet_m;fjet_truthJet_eta;fjet_testing_weight_pt;fjet_testing_weight_pt_dR;fjet_clus_pt;fjet_clus_phi;fjet_clus_eta;fjet_clus_E;fjet_Tau1_wta;fjet_Tau2_wta;fjet_Tau3_wta;fjet_Tau21_wta;fjet_Tau32_wta;fjet_C2;fjet_D2;fjet_M2;fjet_N2

[tree]
name=FlatSubstructureJetTree

[id]
number=0001

[config]
delim=;
```

### Comment: truth labels

The argument *truth-label* defines a selection that is given in the configuration file *truth_label.ini*. The following lines give an example of such a file:

```bash
[rel20.7]
common=TMath::Abs(fjet_truthJet_eta)<2.0;fjet_truthJet_pt/1000>350
sig=TMath::Abs(fjet_truth_dRmatched_particle_flavor)==6;TMath::Abs(fjet_truth_dRmatched_particle_dR)<0.75;TMath::Abs(fjet_dRmatched_WZChild1_dR)<0.75;TMath::Abs(fjet_dRmatched_WZChild2_dR)<0.75;TMath::Abs(fjet_dRmatched_topBChild_dR)<0.75
bkg=
weight=fjet_testing_weight_pt

[rel21.0]
common=TMath::Abs(fjet_truthJet_eta)<2.0;fjet_truthJet_pt/1000>350
sig=TMath::Abs(fjet_truth_dRmatched_particle_flavor)==6;TMath::Abs(fjet_truth_dRmatched_particle_dR)<0.75;fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount>=1;fjet_truthJet_m>140000
bkg=
weight=fjet_testing_weight_pt

[josu]
common=TMath::Abs(fjet_truthJet_eta)<2.0;fjet_truthJet_pt/1000>350
sig=TMath::Abs(fjet_truth_dRmatched_particle_flavor)==6;TMath::Abs(fjet_truth_dRmatched_particle_dR)<0.75;TMath::Abs(fjet_truthJet_dRmatched_particle_dR_top_W_matched)<0.75;fjet_ungroomed_truthJet_m/1000>140;fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount>=1;fjet_ungroomed_truthJet_Split23/1000>20
bkg=
weight=fjet_testing_weight_pt

[split23varied]
common=TMath::Abs(fjet_truthJet_eta)<2.0;fjet_truthJet_pt/1000>350
sig=TMath::Abs(fjet_truth_dRmatched_particle_flavor)==6;TMath::Abs(fjet_truth_dRmatched_particle_dR)<0.75;TMath::Abs(fjet_truthJet_dRmatched_particle_dR_top_W_matched)<0.75;fjet_ungroomed_truthJet_m/1000>140;fjet_truthJet_ungroomedParent_GhostBHadronsFinalCount>=1;fjet_ungroomed_truthJet_Split23/1000.>TMath::Exp(3.3-6.98e-04*fjet_ungroomed_truthJet_pt/1000.)
bkg=
weight=fjet_testing_weight_pt

[incl]
common=TMath::Abs(fjet_truthJet_eta)<2.0;fjet_truthJet_pt/1000>350
sig=TMath::Abs(fjet_truth_dRmatched_particle_flavor)==6;TMath::Abs(fjet_truth_dRmatched_particle_dR)<0.75
bkg=
weight=fjet_testing_weight_pt_dR
```

The *train* section is applied to signal and background. If you use a different definition of the signal, you have to modify those files according to your needs. **Caution: The semicolon is used as a deliminator that separates the individual cuts.**

If you want to convince your self that the correct cuts have been applied to the data set, just inspect the TTree::GetUserInfo (for ROOT file) or h5py.attrs (for HDF5 file NOT YET INCLUDED). For instance, give the part to your fresh generated data set, just do:


```python
import ROOT
t = ROOT.TFile("path/to/root/file.root", "READ").Get("train")
info = {obj.GetName():obj.GetTitle() for obj in t.GetUserInfo()}
print(info)
```


### Comment: training cuts

All events in the output files of this script will fulfill/pass the training cuts shown in section *train* of the previous configuration file; therefore, only such events are used for the training of the DNN solely. However, for the evaluation of the DNN-tagger for the entire data set – including those events that do not fulfill the training cuts –, all events are used. For the evaluation of the tagger, all events that do not pass the training cuts are **a priory** classified as being background events. This is done completely automatically in the script *predict.py* by recovering the training cuts, which have been used in this script, that are saved as metadata in the generated training samples as an attribute to the HDF5 files. Since this is an important point that often gives rise to confusion, let's look at an example. Suppose to generate a data set with the following argument(s) (I'll focus on the cut-train argument for obvious reasons):

```python
fjet_m>40000 && jet_numConstituents>=3
```

The final data set for training will only contain events with a reconstructed jet mass larger than 40 GeV and at least three constituents in the reconstructed jet to ensure, *inter alia*, well-defined substructure variables for the jet. This information of the training cut is stored in the generated HDF5 files an can be accessed via <data set>.attrs of the respective data set. This information will be read later when the data is evaluated with the trained DNN-tagger in the script *add_dnnScore_branch.py*. In this script, each event gets a new variable called *fjet_DnnScore* that gives the classification probability of the tagger for the respective event under consideration. Here the previously used training cuts come into play again: <span style="color:red">all events with fjet_m=<40000 and fjet_numConstituents<3 get a "DNN score" fjet_DnnScore=0 withot being evaluated with the DNN-tagger at all</span> while all other obtain their classification probability within [0,1] from the output of the DNN.



# !Under Construction!
