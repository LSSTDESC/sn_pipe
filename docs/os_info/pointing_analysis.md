#  $${\color{red} OS \space pointing \space analysis}$$

## $${\color{blue} Nightly \space DDF \space pointings \space }$$

- use the script [plot_scripts/os_info/plot_ddf_nightly.py](usage_plot_nightly.md)

## $${\color{blue} Global \space pointing \space analysis}$$

### analysis of the data -> create outputfile with processed data
- use the script [run_scripts/metrics/stat_DD.py](usage_stat_DD.md)
- @ccin2p3, it is possible to use the [anaos.sh](anaos.sh) script using
sh sh_scripts/srun_cc.sh anaos.sh

### plot the data

- use the script [plot_scripts/metrics/plot_DD_stat.py](usage_plot_DD.md)

## $${\color{blue} Nightly \space pointing \space analysis (visits)}$$

### analysis of the data -> create outputfile with processed data
- use the script [run_scripts/os_info/ddf_visits_night.py](usage_process_ddf_visits.md)

### plot the data

- use the script [plot_scripts/os_info/plot_ddf_visits_night.py](usage_plot_ddf_visits.md)

##  $${\color{blue} Nightly \space filter \space allocation \space for \space DDFs}$$

- use the script [plot_scripts/os_info/plot_ddf_nightly.py](usage_plot_ddf_nightly.md)

##  $${\color{blue}  Calibration \space requirements \space (PZ, WL) \space and \space of \space AGN \space constraints}$$

### data analysis -> create outputfile with processed data

- use the script [run_scripts/os_info/nvisits_m5.py](usage_nvisits_m5.md)

- @ccin2p3, it is possible to use the [nvisits_m5.sh](nvisits_m5.sh) script using
sh sh_scripts/srun_cc.sh nvisits_m5.sh

### plot the results

- use the script [plot_scripts/os_info/pz_wl_calib_reqs.py](usage_calib.md)