# sn_pipe

A framework to run the Survey Strategy Support pipeline.

```
This software was developed within the LSST DESC using LSST DESC resources, and so meets the criteria 
given in, and is bound by, the LSST DESC Publication Policy for being a "DESC product". 
We welcome requests to access code for non-DESC use; if you wish to use the code outside DESC 
please contact the developers.

```
## Release Status

This code is under development and has not yet been released.



## Feedback, License etc

If you have comments, suggestions or questions, please [write us an issue](https://github.com/LSSTDESC/sn_pipe/issues).

This is open source software, available for re-use under the modified BSD license.

```
Copyright (c) 2019, the sn_pipe contributors on GitHub, https://github.com/LSSTDESC/sn_pipe/graphs/contributors.
All rights reserved.
```



## **Instruction for installation**

### Getting the package from github
```
 git clone -b thebranch https://github.com/lsstdesc/sn_pipe
 ```
 where thebranch is the branch you would like to use (main, dev, dev_stable, ...)

### Environnement setup
 - The pipeline uses lsst_sim package
 - cernvmfs may be used to have some lsst_sim releases available.
 - to install cvmfs: https://sw.lsst.eu/installation.html
 - ls /cvmfs/sw.lsst.eu/linux-x86_64/lsst_sims/ -> provides a list of lsst_sim releases available.
 - The "setup" is done this way:
 ```
   - on Linux: source setup_release.sh Linux
   - on Mac: source setup_release.sh Mac
```
- The release (automatically) chosen may be found in the current_release.yaml file.
 
**Important : you have to make this setup prior to any operation described below**

###  Installation

The Survey Strategy Support pipeline is supposed to be modular, in the sense that only needed git packages are installed for a given task. The current tasks that may be run are:
- supernovae metrics using the sn_metrics package
- light curve simulation using the sn_simulation package

| Task | package | command for installation|
|----|----|----|
| SN metrics | sn_metrics | python install_sn_pack.py --package=metrics --gitbranch=thebranch|
| LC simulations | sn_simulation|python install_sn_pack.py --package=simulation --gitbranch=thebranch|

## **Running the metrics**

### Installing requested packages
- pip install . --user --install-option="--package=metrics" --install-option="--branch=thebranch"

### Running the Cadence metric
- a notebook illustrating how to run the metric is available in the notebook directory (SNCadence.ipynb) of sn_pipe
- Command line:
  - python run_scripts/run_cadence_metric.py input/param_cadence_metric.yaml
  - A description of the input yaml file is given [here](doc/yaml_cadence.md)
  - you may have to change the 'filename' parameter to the OpSim db name you would like to use as input.
- output : a set of plots: 
- Mean cadence vs mean m5 (5-sigma depth) <img src="doc/cadence_m5_r.png" height="24">
- Histogram of redshift limits <img src="doc/zlim_r.png" height="24">

### Running the Signal-to-Noise Ratio (SNR) metric
- a notebook illustrating how to run the metric is available in the notebook directory (SNSNR.ipynb) of sn_pipe
- Command line:
  -  python run_scripts/run_snr_metric.py input/param_snr_metric.yaml
  - A description of the input yaml file is given [here](doc/yaml_snr.md)
  - you may have to change the 'filename' parameter to the OpSim db name you would like to use as input.
 - output : a set of plots:
   - SNR vs Time (per band and per season) <img src="doc/snr_z_season_1.png" height="24">

## **Running the simulations**

### Installing requested packages
- pip install . --user --install-option="--package=simulation" --install-option="--branch=thebranch"

### Light Curves simulation
- a notebook illustrating how to run the simulation and vizualized outputs is available in the notebook directory (SNSimulation.ipynb) of sn_pipe
- command line:
  - python run_scripts/run_simulation.py input/param_simulation.yaml
- output: two files, hdf5 format:
  - Simu_*.hdf5: astropy table with the list of parameters used for simulation
  - LC*.hdf5: list of (astropy tables) light curves. Each table is composed of metadata (simulation parameters) and of a table with LC points.  
