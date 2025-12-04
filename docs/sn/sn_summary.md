# Data processing

## Usage: run_scripts/sn_analysis/nsn_ddf.py [options]

<pre>
Script to estimate SN - DDF after selection

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location dir[../Output_SN_DD_sigmaInt_0.0_Hounsell_
                        z_smflux_notelrot_G10_JLA]
  --dbList=DBLIST       OS DD list[ddf_list.csv]
  --norm_factor=NORM_FACTOR
                        normalization factor [30]
  --runType=RUNTYPE     run type  [spectroz]
  --timescale=TIMESCALE
                        timescale of the files to process [year]
  --timeslots=TIMESLOTS
                        time slot (season or year) to process [1-10]
  --dataType=DATATYPE   data type [DataFrame]
  --nside=NSIDE         nside healpix parameter [128]
  --outDir=OUTDIR       output dir for the file to store [../sn_summary_ddf]
  --fileName=FILENAME   sn file name to draw [sn_summary_ddf.hdf5]

</pre>

## Run @ccin2p3 (interactive/batch)

Usage: for_batch/scripts/nsn_pixels/nsn_dd.py [options]

<pre>

Script to estimate nsn for WFD - in batch/interactive

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location dir[/sps/lsst/users/gris/Output_SN_DD_sigm
                        aInt_0.0_Hounsell_z_smflux_notelrot_airmass_G10_JLA]
  --dbList=DBLIST       list of OS to process [list_dd.csv]
  --outDir=OUTDIR       output dir [/sps/lsst/users/gris/sn_dd_airmass]
  --scriptName=SCRIPTNAME
                        output sh script [nsn_dd.sh]
  --scriptDir=SCRIPTDIR
                        output sh script dir [sh_scripts]
  --runIt=RUNIT         to run the sh script [1]
  --runMode=RUNMODE     run mode (interactive/batch) [interactive]

</pre>

# Make figures

## Usage: plot_scripts/sn/sn_analyzer_sel_ddf.py [options]

<pre>

Script to analyze SN - DDF after selection

Options:
  -h, --help            show this help message and exit
  --config=CONFIG       OS DD list[input/plots/config_ana.csv]
  --plots=PLOTS         plots to draw [nsn_all,nsn_ud,survey_area]
  --print_nsn=PRINT_NSN
                        to print nsn as a latex table [0]
  --ud_fields=UD_FIELDS
                        UD fields to consider [COSMOS,XMM-LSS]
  --dd_fields=DD_FIELDS
                        DD fields to consider [CDFS,ELAISS1,EDFS_a,EDFS_b]
  --inputDir=INPUTDIR   input dir for the file to draw [../sn_dd_airmass]
  --fileName=FILENAME   sn file name to draw [sn_summary_ddf.hdf5]
  --ref_os=REF_OS       ref OS to normalize the results [None]
  --get_info=GET_INFO   to estimate infos (weather_impact)[None]


</pre>

## Example

### python plot_scripts/sn/sn_analyzer_sel_ddf.py --config=[config_ana_selplot.csv](config_ana_selplot.csv)

[sum(NSN) vs year - all DDFs](plota1.png)

[cumulative sum(NSN) vs year - all DDFs](plota2.png)

[sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - all DDFs](plota3.png)

[cumulative sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - all DDFs](plota4.png)

[Ratio  NSN(z>=0.8, sigmaC<=0.04)/NSN(z>=0.8) vs year - all DDFs](plota5.png)

[sum(NSN) vs year - COSMOS+XMM-LSS](plota6.png)

[cumulative sum(NSN) vs year - COSMOS+XMM-LSS](plota7.png)

[sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - COSMOS+XMM-LSS](plota8.png)

[cumulative sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - COSMOS+XMM-LSS](plota9.png)

[Ratio  NSN(z>=0.8, sigmaC<=0.04)/NSN(z>=0.8) vs year - COSMOS+XMM-LSS](plota10.png)

[NSN/deg2 (z>=0.8, sigmaC<=0.04)/NSN(z>=0.8) vs year - COSMOS+XMM-LSS](plota11.png)


### the option --print_nsn=1 lead to the printing of (nsn, err_nsn) for each OS/year in latex format.