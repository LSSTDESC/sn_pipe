# Light curve fit in sn_pipe

## Install the fit package
```
python pip_sn_pack.py --action install --package sn_fit_lc
```

## Run the fit

It may be done through:
  - the script [run_sn_fit.py](usage_run_sn_fit.md)
  - the following notebook: SNFitLC.ipynb (notebooks directory)
## Output

If the processing was uneventful, two files should be available in the output directory: prodid.yaml and Fit_prodid_fitter.hdf5
   -  prodid.yaml: configuration file for the processing
   - Fit_prodid_fitter_num.hdf5: astropy Table of the result of the fit. 'fitter' is the name of the fitter. The list of available data is the following:

|Dec|RA|color|dL|daymax|ebvofMW|epsilon_color|epsilon_daymax|epsilon_x0|epsilon_x1|healpixID|pixDec|pixRA|ptime|season|snr_fluxsec_meth|status|survey_area|x0|x1|z|z_fit|Cov_t0t0|Cov_t0x0|Cov_t0x1|Cov_t0color|Cov_x0x0|Cov_x0x1|Cov_x0color|Cov_x1x1|Cov_x1color|Cov_colorcolor|t0_fit|x0_fit|x1_fit|color_fit|mbfit|fitstatus|phase_min|phase_max|N_bef|N_aft|N_bef_u|N_aft_u|SNR_u|N_bef_g|N_aft_g|SNR_g|N_bef_r|N_aft_r|SNR_r|N_bef_i|N_aft_i|SNR_i|N_bef_z|N_aft_z|SNR_z|N_bef_y|N_aft_y|SNR_y|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Display result

It is possible to display some of the results of the fit using the sn_plotters package that can be installed using:

```
python pip_sn_pack.py --action install --package sn_plotters
```

Some plots may be obtained using [plot_lcfit.py](../Plots/usage_plot_lcfit.md). The notebook SNFitLC.ipynb also illustrate how to access the file output of the fitting procedure and how to make plots.