# sn_pipe

A framework to run the Survey Strategy Support pipeline.

## **Instruction for installation**

### Getting the package from github

 git clone -b thebranch https://github.com/lsstdesc/sn_pipe
 
 where thebranch is the branch you would like to use (main, dev, dev_stable, ...)

### Environnement setup
- setup script: setup_release.sh
- This script requires up to two arguments and can be run:
  - @NERSC: source setup_release.sh NERSC
  - @CCIN2P3: source setup_release.sh CCIN2P3
- if you wish to run elsewhere then you need to provide the full path to the setup script corresponding to a release including the lsst_sims package, ie source setup_release.sh MYENV full_path_to_setup_script_stack.

**Important : you have to make this setup prior to any operations described below**

## **Running the metrics**

### Installing requested packages
- pip install . --user --install-option="--package=metrics" --install-option="--branch=thebranch" -r requirements.txt

### Running the Cadence metric
- python run_scripts/run_cadence_metric.py input/param_cadence_metric.yaml
- A description of the input yaml file is given [here](doc/yaml_cadence.md)
- you may have to change the 'filename' parameter to the OpSim db name you would like to use as input.
- output : a set of plots: 
- Mean cadence vs mean m5 (5-sigma depth) <img src="doc/cadence_m5_r.png" height="24">
- Histogram of redshift limits <img src="doc/zlim_r.png" height="24">

### Running the Signal-to-Noise Ratio (SNR) metric
-  python run_scripts/run_snr_metric.py input/param_snr_metric.yaml
- A description of the input yaml file is given [here](doc/yaml_snr.md)
- you may have to change the 'filename' parameter to the OpSim db name you would like to use as input.
- output : a set of plots:
   - SNR vs Time (per band and per season) <img src="doc/snr_z_season_1.png" height="24">

