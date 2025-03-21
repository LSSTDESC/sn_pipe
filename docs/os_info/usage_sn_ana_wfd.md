## script plot_scripts/sn/sn_analyzer_sel_wfd.py

## Usage: sn_analyzer_sel_wfd.py [options]
<pre>
Script to analyze SN prod after selection

Options:
  -h, --help            show this help message and exit
  --config=CONFIG       OS list[input/plots/config_ana.csv]
  --dbDir=DBDIR         OS location
                        dir[../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA]
  --norm_factor=NORM_FACTOR
                        Normalization factor [10]
  --runType=RUNTYPE     run type  [spectroz_nosat]
  --timescale=TIMESCALE
                        timescale of the files to process [year]
  --timeslots=TIMESLOTS
                        time slot (season or year) to process [1-10]
  --dataType=DATATYPE   data type [DataFrame]
  --plots=PLOTS         plots to draw [summary,mollweid,density,density_indiv,
                        density_season]
  --outDir=OUTDIR       output dir [../sn_wfd]
  --outName=OUTNAME     output name [nsn_wfd.hdf5]
  --nside=NSIDE         healpix nside parameter [64]
  --vartoplot=VARTOPLOT
                        var to plot (nsn/nsn_cosmo) [nsn]

</pre>