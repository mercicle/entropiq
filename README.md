# Quantum Computing Final Project

Welcome to my QC Project 2 code base!

# Learning DWave

Resources available in Leap: https://cloud.dwavesys.com/leap/resources
Ocean Documentation:  https://docs.ocean.dwavesys.com/en/stable/getting_started.html#gs
Ocean Install Videos:  https://www.youtube.com/playlist?list=PLPvKnT7dgEsuhec_yW9oxcZ6gg9SfRFFV
Ocean Code Examples:  https://github.com/dwave-examples
Introduction to D-Wave and Quantum Annealing: https://docs.dwavesys.com/docs/latest/doc_getting_started.html
Problem Solving Handbook: https://docs.dwavesys.com/docs/latest/doc_handbook.html


# Repository Structure

The helpers functions in /helpers has the bond_parser and jaccard_sim functions created for my analysis.

```
./helpers
./in-data
./out-data
```

The scratchwork has my testing scipts for writing the bond_parser and also working with and scraping/transforming the .mol2 files.

```
./scratchwork
./screenshots
```

This pipeline is the main pipeline. Note, for Step2 forward you must also signup for and retrieve your DWave API token and enter it via the command line with:

```
$ pip install dwave-ocean-sdk
$ dwave config create

```
This .config file is located at:

```
nano /Users/mercicle/Library/Application\ Support/dwave/dwave.conf
```

This is the main pipeline.

```
step0-setup.py
step1-raw-data-ingestion.py
step2-dwave-and-classical-lib-experiments.py
step3-compute-comparison-full-data.py
step4-full-dwave-and-classical-testing.py
step5-full-recalc-classical.py
```


Setup: Installation of necessary Python libraries.

Raw Data Ingestion: programmatically unzipping all .tar.gz files and then all .mol2 files within the target sub-directories.

DWave and Classical Graph Library Experiments: experimentation with both DWave and graph libraries, computing MIS, and visualizing the graphs with MIS encodings.

Create the Full Atom and Bond Data Sets: Transforming the .mol2 text files into atom and atomic bond data frames.

Conduct Full DWave vs. Classical Testing on Full Data: Iterating through the atomic bonds data frame and computing MIS with both DWave and classical library, consolidating the performance results, visualizing the MIS nodes on network visualizations, and conducting the distributional analysis of the Jaccard Similarity and run times.  

Step 5 can be disregarded, I was going to attempt to recalculate based on different MIS library but will just make the previous codes more flexible in the future.

# Data

The in-data folder is git ignored so to download and replicate you must run the following while in in-data directory.

```
wget --base=http://blaster.docking.org/dud/r2/ -i - <<+
ace.tar.gz
ache.tar.gz
ada.tar.gz
alr2.tar.gz
ampc.tar.gz
ar.tar.gz
cdk2.tar.gz
comt.tar.gz
cox1.tar.gz
cox2.tar.gz
dhfr.tar.gz
egfr.tar.gz
er_agonist.tar.gz
er_antagonist.tar.gz
fgfr1.tar.gz
fxa.tar.gz
gart.tar.gz
gpb.tar.gz
gr.tar.gz
hivpr.tar.gz
hivrt.tar.gz
hmga.tar.gz
hsp90.tar.gz
inha.tar.gz
mr.tar.gz
na.tar.gz
p38.tar.gz
parp.tar.gz
pde5.tar.gz
pdgfrb.tar.gz
pnp.tar.gz
ppar.tar.gz
pr.tar.gz
rxr.tar.gz
sahh.tar.gz
src.tar.gz
thrombin.tar.gz
tk.tar.gz
trypsin.tar.gz
vegfr2.tar.gz
energies.tar.gz
+
```

Then, you have to run these two scripts to unzip the tar files AND unzip the .mol .tar files from the unzipped tar files.

```
step0-setup.py
step1-raw-data-ingestion.py
```
