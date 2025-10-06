## Usage: plot_scripts/os_info/plot_os_pixel_info.py [options]
<pre>
Script to plot pixel level OS infos

Options:
   -h, --help            show this help message and exit
  --dbName=DBNAME       dbName to process [test_newb]
  --dbDir=DBDIR         dbDir of the OS to process [../test_metric]
  --nside=NSIDE         healpix nside parameter [128]
  --plots=PLOTS         plots to show [gen_plots,mollview]
  --mollview_seasons=MOLLVIEW_SEASONS
                        plots to show [1-5]
  --fields=FIELDS       plots to show [COSMOS,XMM-
                        LSS,CDFS,ELAISS1,EDFS_a,EDFS_b]
  --fieldType=FIELDTYPE
                        type of field to process (DD/WFD) [DD]
  --timescale=TIMESCALE
                        time scale for the plots (year/season) [year]
  --mollview_var=MOLLVIEW_VAR
                        var to plot in Mollview [cadence,nvisits]
  --gen_var=GEN_VAR     gen var to plot
                        [cadence_year,nvisits_year,cadence_dist]
  --hist_var=HIST_VAR   hist var to plot [nvisits_10yrs]			
</pre>

## example

python plot_scripts/os_info/plot_os_pixel_info.py --dbDir=../dd_pixels --plots=gen_plots --dbName=ddf_dither_0.8_v5.0.0_10yrs --fields=COSMOS --gen_var=cadence_dist,nvisits_dist

[cadence vs distance to the center of pixel set](plotd1.png)

[nvisits vs distance to the center of pixel set](plotd2.png)

python plot_scripts/os_info/plot_os_pixel_info.py --dbDir=../wfd_pixels --dbName=baseline_v5.0.0_10yrs --fieldType=WFD --mollview_season=1 --plots=hist,mollview --nside=64

[Nvisits 10 yrs - WFD - histogram](plotd3.png)

[cadence all years - WFD - histogram](plotd6.png)

[Nvisits year 1 - WFD - Mollweid view](plotd4.png)

[cadence year 1- WFD - Mollweid view](plotd5.png)