## UD seasons features

### process the data

<pre>
Usage: run_scripts/os_info/ddf_night_os_stat.py [options]

Script to analyse DDF visits on a nightly basis from pointings

Options:
  -h, --help       show this help message and exit
  --dbDir=DBDIR    file directory [../ddf_visits_night]
  --dbName=DBNAME  OS to process [baseline_v5.3.0_10yrs]
  --outDir=OUTDIR  output directory [../ddf_visits_night_ud]

</pre>

### figures of the results

<pre>
Usage: plot_scripts/os_info/plot_ddf_night_os_stat.py [options]

Script to analyse DDF visits on a nightly basis from pointings

Options:
  -h, --help       show this help message and exit
  --dbDir=DBDIR    file directory [../ddf_visits_night_ud]
  --dbName=DBNAME  OS name [baseline_v5.3.0_10yrs]

</pre>

[Season length per season for all DDFs](season_length_all_ddf.png)

[Season length for UD seasons for all DDFs](season_length_ud_all_ddf.png)

[Nvisits per filter/season for COSMOS](nvisits_filter_season_cosmos.png)

[Delta time start season/start UD](deltat_start_ud.png)

[Delta time end season/end UD](deltat_end_ud.png)