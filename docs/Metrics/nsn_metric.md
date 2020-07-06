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
<ul>
<li> using python script [run_metrics.py](usage_run_metrics.md)
</ul>

## Output analysis