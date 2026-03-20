# How to generate (and display) SN Ia flux and SED vs time

## Usage: run_scripts/simulation/run_flux_spectra_sn.py [options]

<pre>
script to generate LC and spectra for SNe Ia

Options:
  -h, --help            show this help message and exit
  --x1=X1               SN Ia strech [0.0]
  --color=COLOR         SN Ia color [0.0]
  --daymax=DAYMAX       SN Ia T0 [68000]
  --z=Z                 SN Ia redshift [0.8]
  --ebvofMW=EBVOFMW     E(B-V) of MW [0.01]
  --airmass=AIRMASS     airmass [1.2]
  --pwv=PWV             precipitable water vapor [mm] [4.0]
  --ozone=OZONE         ozone [dobson] [300.0]
  --aerosol=AEROSOL     aerosol value  [0.01]
  --sed=SED             to estimate sn sed [0]
  --outDir=OUTDIR       output directory [../sn_flux_spectra]
  --outName=OUTNAME     output file name [simu1]
  --outDir_display=OUTDIR_DISPLAY
                        output dir for SN displays [None]

</pre>

## Example

python run_scripts/simulation/run_flux_spectra_sn.py --sed=1

should lead to the generation of 2 files in the default output directory (../sn_flux_spectra):
- sn_flux_simu1.hdf5: SN Ia light curves
- sn_sed_simu1.hdf5: SN Ia SED

## Display the results

Usage: plot_scripts/sn/plot_flux_sed.py [options]

<pre>
script to plot LC and spectra for SNe Ia

Options:
  -h, --help           show this help message and exit
  --dataDir=DATADIR    data dir [../sn_flux_spectra]
  --fluxFile=FLUXFILE  flux file [sn_flux_simu1.hdf5]
  --sedFile=SEDFILE    sed file [sn_sed_simu1.hdf5]
  --phases=PHASES      phases to plot [-10,0,20]

</pre>

[SED and fluxes vs phase](fig1.png)