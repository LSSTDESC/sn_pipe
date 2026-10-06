#for a set of strategies

## Usage: plot_scripts/os_info/plot_os_pixel_diff.py
<pre>
Script to plot pixel level OS infos

Options:
  -h, --help            show this help message and exit
  --dbList=DBLIST       dbList to process [list_OS.csv]
  --dbDir=DBDIR         dbDir of the OS to process [../wfd_pixels_5.3_new]
  --fieldType=FIELDTYPE
                        field type to process [WFD]
  --ref_OS=REF_OS       ref OS [baseline_v5.3.0_10yrs]
  --nvisits_10yrs_min=NVISITS_10YRS_MIN
                        min nvisits after 10 yrs  [500]
  --nvisits_10yrs_max=NVISITS_10YRS_MAX
                        max nvisits after 10 yrs  [2000]
  --ebvofMW_max=EBVOFMW_MAX
                        max E(B-V) [0.25]
  --nside=NSIDE         healpix nside parameter [64]

</pre>

[$\Delta$ cadence](summary_delta_cadence_shrink.png)
[relative cadence](summary_rel_cadence_shrink.png)
[$\Delta$ Nvisits](summary_delta_nvisits_shrink.png)
[relative survey area](summary_rel_area_shrink.png)
[survey area](summary_area_shrink.png)