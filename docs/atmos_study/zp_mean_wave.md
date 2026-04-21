## Data production

### Usage: run_scripts/telescope/zero_points_atmos.py [options]

<pre>

Script to estimate zp and mean_wave from atmos parameters

Options:
  -h, --help            show this help message and exit
  --telDir=TELDIR       tel main dir [throughputs]
  --throughDir=THROUGHDIR
                        throughput dir [baseline]
  --tag=TAG             tag version of the throughputs [1.9]
  --airmass_min=AIRMASS_MIN
                        airmass min value [1.0]
  --airmass_max=AIRMASS_MAX
                        airmass max value [2.5]
  --airmass_step=AIRMASS_STEP
                        airmass step value [0.0]
  --sigma_airmass_min=SIGMA_AIRMASS_MIN
                        sigma airmass min value [0.0]
  --sigma_airmass_max=SIGMA_AIRMASS_MAX
                        sigma airmass max value [0.02]
  --sigma_airmass_step=SIGMA_AIRMASS_STEP
                        sigma airmass step value [0.0]
  --pwv_min=PWV_MIN     pwv min value [5.0]
  --pwv_max=PWV_MAX     pwv max value [7.0]
  --pwv_step=PWV_STEP   pwv step value [0.0]
  --sigma_pwv_min=SIGMA_PWV_MIN
                        sigma pwv min value [0.0]
  --sigma_pwv_max=SIGMA_PWV_MAX
                        sigma pwv max value [0.3]
  --sigma_pwv_step=SIGMA_PWV_STEP
                        sigma pwv step value [0.0]
  --ozone_min=OZONE_MIN
                        ozone min value [330.0]
  --ozone_max=OZONE_MAX
                        ozone max value [300.0]
  --ozone_step=OZONE_STEP
                        ozone step value [0.0]
  --sigma_ozone_min=SIGMA_OZONE_MIN
                        sigma ozone min value [0.0]
  --sigma_ozone_max=SIGMA_OZONE_MAX
                        sigma ozone max value [20.0]
  --sigma_ozone_step=SIGMA_OZONE_STEP
                        sigma ozone step value [0.0]
  --aerosol_min=AEROSOL_MIN
                        aerosol min value [0.05]
  --aerosol_max=AEROSOL_MAX
                        aerosol max value [0.1]
  --aerosol_step=AEROSOL_STEP
                        aerosol step value [0.0]
  --sigma_aerosol_min=SIGMA_AEROSOL_MIN
                        sigma aerosol min value [0.0]
  --sigma_aerosol_max=SIGMA_AEROSOL_MAX
                        sigma aerosol max value [0.02]
  --sigma_aerosol_step=SIGMA_AEROSOL_STEP
                        sigma aerosol step value [0.0]
  --beta_min=BETA_MIN   beta min value [1.4]
  --beta_max=BETA_MAX   beta max value [0.4]
  --beta_step=BETA_STEP
                        beta step value [0.0]
  --sigma_beta_min=SIGMA_BETA_MIN
                        sigma beta min value [0.0]
  --sigma_beta_max=SIGMA_BETA_MAX
                        sigma beta max value [0.4]
  --sigma_beta_step=SIGMA_BETA_STEP
                        sigma beta step value [0.0]
  --outDir=OUTDIR       outputDir [../zp_atmos]
  --outName=OUTNAME     outName [zp_atmos_config1.hdf5]
  --param_outDir=PARAM_OUTDIR
                        output dir for params [None]
  --save_random_dir=SAVE_RANDOM_DIR
                        outputDir for random trials ['']
  --ntrial=NTRIAL       number of trials [1]
  --nsample=NSAMPLE     number of sample per trial [1000]
  --nproc=NPROC         number of procs [8]

</pre>

An example on how to use this script is available: see sh_scripts/zp_atmos.sh

## Analyse and make figure

### Usage: plot_scripts/zp_wave_atmos/plot_sigma_zp_wave.py [options]

<pre>
analyze and plot zp and mean wave

Options:
  -h, --help            show this help message and exit
  --dataDir=DATADIR     data dir [../zp_atmos]
  --atmos_param=ATMOS_PARAM
                        atmospheric parameter[ozone]
  --band=BAND           band to plot [y]
  --plots=PLOTS         what to plot [map,sigma]

</pre>

[grid of sigma_zp_y vs airmass](map_sigma_mean_wave_y_airmass.png)

[grid of sigma_mean_wave_y vs airmass](map_sigma_mean_wave_y_airmass.png)

[sigma_zp_y vs sigma_pwv](sigma_pz_sigma_pwv.png)

[sigma_mean_wave_y vs sigma_pwv](sigma_mean_wave_sigma_pwv.png)

### Usage: run_scripts/telescope/fit_sigmas_zp_wave_vs_sigma_atmos.py [options]

<pre>

Fit sigma_zp and sigma_mean_wave vs sigma of atmos params

Options:
  -h, --help            show this help message and exit
  --dataDir=DATADIR     data dir [../zp_atmos]
  --atmos_param=ATMOS_PARAM
                        atmospheric parameters [airmass,ozone,aerosol,pwv]
  --plots=PLOTS         plots [vs_airmass,summary,from_sigmas]
  --sigmas=SIGMAS       sigmas of atmos params [3e-3,20,5e-3,0.2]
  --unit=UNIT           unit of sigmas of atmos params [,DU,,mm]

</pre>

[sigma_zp vs sigma_pwv](fig4a.png)

[sigma_mean_wave vs sigma_pwv](fig3a.png)

[sigma_zp vs band and atmos. param for a set of atmos. param sigmas](fig1a.png)

[sigma_zp budget vs band](fig2a.png)

### Usage: run_scripts/telescope/get_zp_wave.py

<pre>
Estimate sigma_zp_tot from config and sigma_atmos from sigma_zp_atmos

Options:
  -h, --help            show this help message and exit
  --dataDir=DATADIR     data dir [../zp_atmos]
  --atmos_param=ATMOS_PARAM
                        atmospheric parameters [airmass,ozone,aerosol,pwv]
  --config=CONFIG       sigma atmos parameters [config_atmos.csv]
  --plots=PLOTS         plots to perform [from_config,sigmas]
  --plotDir=PLOTDIR     where the plots will be saved [../iso_zp]

</pre>

[sigma_zp_tot vs filter](iso_zp/summary_zp.png)

[sigma_airmass vs filter for sigma_zp_airmass values](iso_zp/sigma_airmass_0.png)

[sigma_airmass/airmass vs filter for sigma_zp_airmass values](iso_zp/sigma_airmass_1.png)

[sigma_ozone vs filter for sigma_zp_ozone values](iso_zp/sigma_ozone_0.png)

[sigma_ozone/airmass vs filter for sigma_zp_ozone values](iso_zp/sigma_ozone_1.png)

[sigma_aerosol vs filter for sigma_zp_aerosol values](iso_zp/sigma_aerosol_0.png)

[sigma_aerosol/aerosol vs filter for sigma_zp_aerosol values](iso_zp/sigma_aerosol_1.png)

[sigma_pwv vs filter for sigma_zp_pwv values](iso_zp/sigma_pwv_0.png)

[sigma_pwv/pwv vs filter for sigma_zp_pwv values](iso_zp/sigma_pwv_1.png)