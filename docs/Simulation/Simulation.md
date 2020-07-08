# Light curves simulation in sn_pipe

## General comments

The simulation package of sn_pipe (sn_simulation) aims at generating supernovae light curves from a set of observations (coming for instance from Observing Strategies) and a set of parameters (type of simulator, cosmology, supernovae parameters, ...), both being chosen by the user. At the moment:
 - two simulators are avalaible: sn_cosmo and sn_fast
 - only type 1a supernovae can be generated

The parameters for the simulation are part of  [yaml file](../../input/simulation/param_simulation_gen.yaml) ingested by the scripts used for the simulation.
 


## Installation of the simulation package

```
python pip_sn_pack.py --action install --package=sn_simulation
```

## Simulating light curves for supernovae

There are currently two ways to simulate light curves in sn_pipe:
   - using python scripts
   - using a notebook


## Analyzing simulation output



