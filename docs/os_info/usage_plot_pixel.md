# per OS

## Usage: plot_scripts/os_info/plot_os_pixel_info.py [options]
<pre>
Script to plot pixel level OS infos

Options:
  -h, --help            show this help message and exit
  --dbName=DBNAME       dbName to process [test_newb]
  --dbDir=DBDIR         dbDir of the OS to process [../test_metric]
  --nside=NSIDE         healpix nside parameter [128]
  --plots=PLOTS         plots to show [gen_plots,mollview]
  --seasons=SEASONS     seasons to show [1-5]
  --fields=FIELDS       fields to show [COSMOS,XMM-
                        LSS,CDFS,ELAISS1,EDFS_a,EDFS_b]
  --fieldType=FIELDTYPE
                        type of field to process (DD/WFD) [DD]
  --timescale=TIMESCALE
                        time scale for the plots (year/season) [year]
  --mollview_var=MOLLVIEW_VAR
                        var to plot in Mollview [cadence,nvisits]
  --gen_var=GEN_VAR     gen var to plot
                        [cadence_year,nvisits_year,cadence_dist,nvisits_dist]
  --hist_var=HIST_VAR   hist var to plot [nvisits_10yrs,cadence_year]
  --mollview_outDir=MOLLVIEW_OUTDIR
                        output dir for mollview figures [None]
  --nvisits_10yrs_min=NVISITS_10YRS_MIN
                        min nvisits after 10 yrs (to remove hot spots) [1200]
		
</pre>

## example

python plot_scripts/os_info/plot_os_pixel_info.py --dbDir=../dd_pixels --plots=gen_plots --dbName=ddf_dither_0.8_v5.0.0_10yrs --fields=COSMOS --gen_var=cadence_dist,nvisits_dist

[cadence vs distance to the center of pixel set](plotd1.png)

[nvisits vs distance to the center of pixel set](plotd2.png)

python plot_scripts/os_info/plot_os_pixel_info.py --dbDir=../wfd_pixels_5.3 --dbName=baseline_v5.3.0_10yrs,comp_survey_v5.3.0_10yrs --fieldType=WFD --seasons=10yrs --plots=mollview,hist --hist_var=nvisits --mollview_var=nvisits --nside=64 --nvisits_10yrs_min=10000

[Nvisits 10 yrs - WFD - histogram](plotd3.png)

[Nvisits 10 years - WFD - Mollweid view - OS1](plotd4.png)

[Nvisits 10 years - WFD - Mollweid view - OS2](plotd5.png)

[diff Nvisits 10 years - WFD - Mollweid view](plotd6.png)

## Summary plots (DDF)

### area ve OS

#### Usage: plot_scripts/os_info/plot_dd_pixels_summary.py [options]

<pre>
Script to plot pixel level OS infos

Options:
  -h, --help            show this help message and exit
  --config=CONFIG       config file [config_ana_selplot_part1.csv]
  --dbDir=DBDIR         data dir [../dd_pixels]
  --fields=FIELDS       fields to show [COSMOS,XMM-
                        LSS,CDFS,ELAISS1,EDFS_a,EDFS_b]
  --timescale=TIMESCALE
                        timescale for the plots (year/season) [year]
  --nside=NSIDE         nside healpix parameter [128]

</pre>

#### Example

[area (per year) vs OS](ploth1.png)

[mean area vs dither parameter](ploth2.png)

### (Nvisits,cadence) vs radius

#### data processing

Usage: run_scripts/os_info/calc_radius_dd_pixels.py [options]
<pre>
Script to plot pixel level OS infos

Options:
  -h, --help         show this help message and exit
  --dbList=DBLIST    dblist to process [list_db.csv]
  --dbDir=DBDIR      dbDir of the OS to process [../dd_pixels]
  --outName=OUTNAME  output file name [data_radius.hdf5]

</pre>

@ccin2p3: use the script [radius_pixel.sh](radius_pixel.sh) using "sh [srun_test.sh](srun_test.sh) radius_pixel.sh"

#### display results

Usage: plot_scripts/os_info/plot_dd_radius.py [options]

<pre>
Script to plot pixel level OS radius

Options:
  -h, --help       show this help message and exit
  --fName=FNAME    file name to process [data_radius.hdf5]
  --config=CONFIG  config file [config_ana_selplot_part2.csv]
</pre>

[Nvisits and cadence vs radius - COSMOS - year 1](plotj1.png)

[Nvisits and cadence vs radius - COSMOS - year 2](plotj2.png)

[Nvisits and cadence vs radius - COSMOS - year 3](plotj3.png)

