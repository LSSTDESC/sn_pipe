## Impact of atmos params uncertainty measurements

### Data production

- see example_scripts/zp_atmos.sh

### Data analysis and figures

#### Usage: plot_scripts/zp_wave_atmos/plot_sigma_zp_wave.py [options]

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

[grid of sigma_zp_y vs airmass](map_sigma_zp_y_airmass.png)

[grid of sigma_mean_wave_y vs airmass](map_sigma_mean_wave_y_airmass.png)

[sigma_zp_y vs sigma_pwv](sigma_pz_sigma_pwv.png)

[sigma_mean_wave_y vs sigma_pwv](sigma_mean_wave_sigma_pwv.png)

#### Usage: run_scripts/telescope/fit_sigmas_zp_wave_vs_sigma_atmos.py [options]

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
  --dirPlot=DIRPLOT     dir for output plots [../plot_atmos]
</pre>

[sigma_zp vs sigma_pwv (grizy)](fig4a.png)

[sigma_zp vs sigma_pwv (gr,ri,iz,zy)](fig_std_zp_sigma_pwv_gr_ri_iz_zy.png)

[sigma_mean_wave vs sigma_pwv](fig3a.png)

[sigma_zp vs band and atmos. param for a set of atmos. param sigmas](from_sigma_zp_sigma.png)

[sigma_zp budget vs band](from_sigma_zp_frac.png)

[summary zp (1mmag)](summary_zp.png)

[summary mean wave (0.1 nm)](summary_mean_wave.png)


#### Usage: run_scripts/telescope/get_zp_wave.py

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