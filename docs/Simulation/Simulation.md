# Light curves simulation in sn_pipe

## General comments

The simulation package of sn_pipe (sn_simulation) aims at generating supernovae light curves from a set of observations (coming for instance from Observing Strategies) and a set of parameters (type of simulator, cosmology, supernovae parameters, ...), both being chosen by the user. At the moment:
 - two simulators are avalaible: sn_cosmo and sn_fast
 - all types of supernovae can be generated

The parameters for the simulation are part of  yaml file (an example: input/simulation/param_simulation_gen.yaml) ingested by the scripts used for the simulation. The definition of the parameters is available [here](yaml_file.md).
 
## Scripts to run the simulation

The scripts are located in the directory run_scripts/simulation

## Installation of the simulation package

```
python pip_sn_pack.py --action install --package=sn_simulation
```

## Simulating light curves for supernovae

There are currently two ways to simulate light curves in sn_pipe:
   - using python scripts
     - a yaml file may be generated using the script run_scripts/make_yaml/make_yaml_simulation.py.
       Use then  [run_simulation_from_yaml.py](../Simulation/usage_run_simulation_yaml.md)
     -  the yaml file est defined from a generic file: use  [run_simulation.py](../Simulation/usage_run_simulation.md) 
   - using a notebook
     - an example is available in notebooks/SNSimulation.ipynb

## Output of the simulation process

If the processing is successful, three files should be available in the output directory (defined in the Output section of the yaml file). These files will have the names prodid.yaml, Simu_prodid.hdf5, LC_prodid.hdf5 where prodid is the 'ProductionID' value defined in the input yaml file:
 - prodid.yaml: parameters of the simulation
 - Simu_prodid.hdf5: set of astropy Table with the list of the simulation parameters per LC. The following columns are accessible:
 
|SNID|index_hdf5|season|fieldname|fieldid|n_lc_points|area|RA|Dec|x0|epsilon_x0|x1|epsilon_x1|color|epsilon_color|daymax|epsilon_daymax|z|survey_area|healpixID|pixRA|pixDec|dL|ptime|snr_fluxsec_meth|status|ebvofMW|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

 - LC_prodid.hdf5 : set of astropy Tables containing the light curve points. Each astropy Table is composed:
   - metadata:

|Dec|RA|color|dL|daymax|ebvofMW|epsilon_color|epsilon_daymax|epsilon_x0|epsilon_x1|healpixID|pixDec|pixRA|ptime|season|snr_fluxsec_meth|status|survey_area|x0|x1|z|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
   - LC points:

|m5|time|exptime|numExposures|band|airmass|sky|moonPhase|seeingFwhmEff|seeingFwhmGeom|filter_cosmo|flux|mag|gamma|flux_e_sec|snr_m5|magerr|fluxerr|zp|zpsys|phase|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Analyzing simulation output

Install the sn_plotters package to visualize output of the simulation

```
python pip_sn_pack.py --action install --package=sn_plotters
```

Use the script [plot_simu.py](../Plots/usage_plot_simu.md) to display simulation results.



