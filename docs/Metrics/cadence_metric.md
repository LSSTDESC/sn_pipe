# Cadence metric

## Definition
This metric is an estimate of the redshift limit z of a faint [(x1,color) = (-2.0,0.2)] supernovae. Depending on Signal-to-Noise thresholds (defined by the user), mean cadences and mean five-sigma depth values (from simulation), z is corresponding to the detection limit. This metric reflects cadences and m5 dependencies on supernova detection.

## Installation of the metric package

```
python pip_sn_pack.py --action install --package=sn_metrics
```

## Input parameters

 - band
 - Signal-To-Noise ratio (SNR per band): use as detection thresholds (typical values given below)
 - mag_range: magnitude range considered
 - dt_range : cadence range (in days-1) for the study
 - zmin, zmax: min and mad redshifts for the study
 - Li_files : list of npy files with light curves
 - mag_to_flux : list of npy files with mag to flux conversion

This metric may be run yearly, per season or using the complete survey.

## How to run this metric

There are currently two ways of running this metric
 - use the script [run_metrics.py](usage_run_metrics.md)
 - use the notebook SNCadence.ipynb located in the notebooks directory.

## Output analysis

The analysis/display of the metric results can be done using the sn_plotters package that can be installed as follow:

```
python pip_sn_pack.py --action install --package=sn_plotters
```

The script [plot_cadence_metric.py](../Plots/usage_plot_cadence_metric.md) may be used to display the results. An illustration of the plots obtained is also available in the above-mentioned notebook.