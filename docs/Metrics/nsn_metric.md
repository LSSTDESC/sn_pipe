# NSN metric

## Definition

The NSN metric gives an estimate of the number of *well-sampled* type Ia supernovae as a function of the redshift limit. The redshift limit (zlim) est defined for faint supernovae (SALT2 parameters: x1=-2.0, color=0.2) and corresponds to a *complete* sample. It is estimated from observation efficiencies as a function of the redshift obtained from fast simulation of light curves (templates). The selection criteria for the faint supernovae are:
<ul>
 <li>Signal-to-Noise min per point (typically: 5)
 <li> a number of LC points before max (typically: 4)
 <li> a number of LC points after max (typically: 5)
 <li> a number of LC points with phase <= (typically: 1)
 <li> a number of LC points with phase >= (typically: 1)
 <li> &sigma;<sub>C</sub> &le; 0.04 where &sigma;<sub>C</sub> is the error on the color parameter of the supernova(estimated from Fisher matrix). 
 </ul>
Once zlim has been estimated, the number of type Ia supernovae is given for medium supernovae (SALT2 parameters: x1=-0.0, color=0.0) with z&le;zlim. Fast simulations are also used (templates) and the same selection criteria are applied at this stage.

## How to run this metric

There are currently two ways to run this metric:
* using python script [run_metrics.py](usage_run_metrics.md)

Output of this metric is composed of two files:

* a .yaml file containing all the parameters used for the processing
* a .hdf5 file containing the results of the metric (astropy tables)


## Output analysis

The analysis of the metric results may be performed using the sn_plotters package. Two types of plots (the scripts are available in plot_scripts/metrics):
* for a given Observing Strategy (OS), plots of zlim, NSN, observing parameters may be obtained using the following script: [plot_nsn_metric_OS.py](../Plots/usage_plot_nsn_metric_OS.md)
* for a set of OS: a summary plot (NSN, zlim) may be obtained using [plot_nsn_metric_DD_summary.py](../Plots/usage_plot_nsn_metric_DD_summary.md) (for DD) and [plot_nsn_metric_WFD_summary.py](../Plots/usage_plot_nsn_metric_WFD_summary.md) (for WFD).

