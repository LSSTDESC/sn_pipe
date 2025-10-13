# WFD summary plots

## Data processing

### Usage: run_scripts/sn_analysis/nsn_wfd.py [options]

<pre>

Script to estimate nsn for WFD

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location
                        dir[../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA]
  --dbList=DBLIST       list of OS to process [list_wfd.csv]
  --norm_factor=NORM_FACTOR
                        Normalization factor [10]
  --runType=RUNTYPE     run type  [spectroz_nosat]
  --timescale=TIMESCALE
                        timescale of the files to process [year]
  --timeslots=TIMESLOTS
                        time slot (season or year) to process [1-10]
  --dataType=DATATYPE   data type [DataFrame]
  --outDir=OUTDIR       output dir [../sn_wfd]
  --outName=OUTNAME     output name [nsn_wfd.hdf5]

</pre>

### Example

python run_scripts/sn_analysis/nsn_wfd.py --dbDir=/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA --dbList=[list_OS_nsn_wfd.csv](list_OS_nsn_wfd.csv) --outDir=/sps/lsst/users/gris/sn_wfd

-> a set of files (nsn_wfd.hdf5) will be produced in the dir /sps/lsst/users/gris/sn_wfd/dbName

### Batch@ccin2p3

It is possible to use the batch system @ccin2p3 for the estimation of nsn.

Usage: for_batch/scripts/nsn_pixels/nsn_wfd.py [options]

<pre>
Script to estimate nsn for WFD - in batch

Options:
  -h, --help       show this help message and exit
  --dbDir=DBDIR    OS location dir[/sps/lsst/users/gris/Output_SN_WFD_sigmaInt
                   _0.0_Hounsell_z_smflux_notelrot_airmass_G10_JLA]
  --dbList=DBLIST  list of OS to process [list_wfd.csv]
  --outDir=OUTDIR  output dir [/sps/lsst/users/gris/sn_wfd_airmass]
  
</pre>

## Plot summary

### Usage: plot_scripts/sn/sn_analyzer_sel_wfd.py [options]

<pre>
Script to analyze SN prod after selection

Options:
  -h, --help            show this help message and exit
  --config=CONFIG       OS list[input/plots/config_ana.csv]
  --dbDir=DBDIR         OS location dir[../sn_wfd]
  --timeslots=TIMESLOTS
                        time slot (season or year) to process [1-10]
  --timescale=TIMESCALE
                        timescale of the files to process [year]
  --plots=PLOTS         plots to draw [summary,mollweid,mollweid_ffmpeg,densit
                        y,density_indiv,density_season]
  --outDir=OUTDIR       output dir [../sn_wfd]
  --fName=FNAME         output name [nsn_wfd.hdf5]
  --nside=NSIDE         healpix nside parameter [64]
  --vartoplot=VARTOPLOT
                        var to plot (nsn/nsn_cosmo) [nsn]
  --OS_ref=OS_REF       ref OS for normalization [None]
  --print_nsn=PRINT_NSN
                        to print nsn as a latex table [0]

</pre>

### Examples

python plot_scripts/sn/sn_analyzer_sel_wfd.py --config=[config_ana_selplot.csv](config_ana_selplot.csv) --plots=summary

[NSN WFD summary plot](plote1.png)

[NSN WFD summary plot - cosmology grade sample](plote2.png)

[Relative NSN (cumul) variation vs year](plote3.png)

python plot_scripts/sn/sn_analyzer_sel_wfd.py --config=config_ana_selplot.csv --plots=mollweid

-> files located in outDir/Os_name

[NSN vs year - Mollweid view](plote4.png)

python plot_scripts/sn/sn_analyzer_sel_wfd.py --config=config_ana_selplot.csv --plots=density

[NSN/deg2 vs Dec slices](plote5.png)

[area vs dec slices](plote6.png)

python plot_scripts/sn/sn_analyzer_sel_wfd.py --config=config_ana_selplot.csv --plots=density_season --vartoplot='nsn_cosmo'

[NSN/deg2 per Dec slices for desc_ddf_gen_0.80_v4.3.1_10yrs vs year](plote7.png)

[NSN/deg2 per Dec slices for baseline_v4.3.1_10yrs vs year](plote8.png)

python plot_scripts/sn/sn_analyzer_sel_wfd.py --config=config_ana_selplot.csv --plots=density_indiv --vartoplot='nsn_cosmo'

[NSN/deg2 per Dec slices for desc_ddf_gen_0.80_sn_v4.3.1_10yrs](plote9.png)

[NSN/deg2 per Dec slices for baseline_v4.3.1_10yrs](plote10.png)

### the option --print_nsn=1 lead to the printing of (nsn, err_nsn) for each OS/year in latex format.