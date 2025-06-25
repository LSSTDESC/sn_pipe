##  Usage plot_scripts/os_info/pz_wl_calib_reqs.py

<pre>
Script to plot calib reqs from PZ and WL

Options:
  -h, --help         show this help message and exit
  --config=CONFIG    OS DD list[config_ana_selplot.csv]
  --dirFile=DIRFILE  file directory [../nvisits_m5]
  --fields=FIELDS    fields to process [DD:COSMOS,DD:ECDFS,DD:XMM_LSS,DD:ELAIS
                     S1,DD:EDFS_a,DD:EDFS_b]
  --plots=PLOTS      plots to draw [plot_global_wl,plot_global_agn,summary_req
                     s_wl,summary_reqs_agn,summary_reqs_pz]

</pre>

## Examples

### python plot_scripts/os_info/pz_wl_calib_reqs.py --fields=DD:ECDFS --plots=plot_global_wl
-> should lead to a set of plots (one per band) as [this one](plotf1.png)

### python plot_scripts/os_info/pz_wl_calib_reqs.py --fields=DD:ECDFS --plots=plot_global_agn
-> should lead to a set of plots (one per band) as [this one](plotf2.png)

### python plot_scripts/os_info/pz_wl_calib_reqs.py --plots=summary_reqs_wl
-> should lead to a set of plots (one per band) as [this one](plotf3.png)

### python plot_scripts/os_info/pz_wl_calib_reqs.py --plots=summary_reqs_agn
-> should lead to a set of plots (one per band) as [this one](plotf4.png)

### python plot_scripts/os_info/pz_wl_calib_reqs.py --plots=summary_reqs_pz
-> should lead to a set of plots (one per band) as [this one](plotf5.png)