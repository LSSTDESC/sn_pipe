## Usage: plot_os_pixel_info.py [options]
<pre>
Script to plot pixel level OS infos

Options:
  -h, --help            show this help message and exit
  --dbName=DBNAME       dbName to process [test_newb]
  --dbDir=DBDIR         dbDir of the OS to process [../test_metric]
  --nside=NSIDE         healpix nside parameter [128]
  --plots=PLOTS         plots to show[cadence_season,nvisits_season,cadence_di
                        st,nvisits_dist,mollview_cadence,mollview_nvisits]
  --mollview_seasons=MOLLVIEW_SEASONS
                        plots to show[1-5]
</pre>

## example

python plot_scripts/os_info/plot_os_pixel_info.py --dbName=baseline_v4.3.1_10yrs --plots=nvisits_dist,cadence_dist will lead to the following plots of the file ../test_metric/baseline_v4.0_10yrs.hdf5.

[cadence vs distance to the center of pixel set](plotd2.png)

[nvisits vs distance to the center of pixel set](plotd2.png)
