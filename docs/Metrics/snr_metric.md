# SNR metric

## Definition
This metric is an estimate of the fraction of faint [(x1,color) = (-2.0,0.2)] supernovae with a Signal-to-Noise Ratio (SNR) higher or equal to the SNR corresponding to a regular cadence. It is defined per band and is a reasonable estimate of the gap effects on SN detection.

Let us suppose that we have a set of measurements of (fluxes,error fluxes): (f<sub>i</sub>,&sigma;<sub>i</sub>). Then the Signal-to-Noise Ratio (SNR) may be written, per band b:

<img src="SNR.png" height="100">

In the background-dominating regime, one has:

<img src="sigma_bd.png" height="100">

where f<sub>i</sub><sup>5,b</sup> is the 5-&sigma; flux related to the 5&sigma; depth (m<sub>5</sub>) by:

<img src="m5_f5.png" height="130">

Since m<sub>5</sub> is given by observing conditions, it is possible to estimate SNR<sup>b>/sup> provided a flux template for supernovae is available.


## Installation of the metric package

```
python pip_sn_pack.py --action install --package=sn_metrics
```

## Input parameters

 - band
 - configuration file for fakes
 - mag_range: magnitude range considered
 - Li_files : list of npy files with light curves
 - mag_to_flux : list of npy files with mag to flux conversion

This metric may be run yearly, per season or using the complete survey.

## How to run this metric

There are currently two ways of running this metric
 - use the script [run_metrics.py](usage_run_metrics.md)
 - use the notebook SNSNR.ipynb located in the notebooks directory.

## Output analysis

The analysis/display of the metric results can be done using the sn_plotters package that can be installed as follow:

```
python pip_sn_pack.py --action install --package=sn_plotters
```

The script [plot_snr_metric.py](../Plots/usage_plot_snr_metric.md) may be used to display the results. An illustration of the plots obtained is also available in the above-mentioned notebook.