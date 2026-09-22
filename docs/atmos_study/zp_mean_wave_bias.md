### Impact of atmos params bias

### Data analysis and grid plots

#### Usage:plot_scripts/zp_wave_atmos/plot_bias_zp_wave.py [options]

<pre>
analyze and plot zp and mean wave from bias run

Options:
  -h, --help            show this help message and exit
  --dataDir=DATADIR     data dir [../zp_atmos_bias]
  --atmos_param=ATMOS_PARAM
                        bias atmospheric parameter [airmass]
  --obs=OBS             variable to plot [zp]
  --bands=BANDS         filters to plot [grizy]
  --outDir=OUTDIR       filters to plot [../zp_atmos_bias_summary]
  --what=WHAT           what to do [grid_plot,interp_estimates]
</pre>

[grid: airmass bias - y band](iso_zp/grid_bias_airmass_y.png)

[grid: airmass bias - r band](iso_zp/grid_bias_airmass_r.png)

[grid: pwv bias - y band](iso_zp/grid_bias_pwv_y.png)

[grid: aerosol bias - g band](iso_zp/grid_bias_aerosol_g.png)

[grid: aerosol bias - r band](iso_zp/grid_bias_aerosol_r.png)

[grid: ozone bias - r band](iso_zp/grid_bias_ozone_r.png)

### Summary plots

#### Usage: plot_scripts/zp_wave_atmos/plot_bias_zp_wave_summary.py

<pre>

summry plots from bias results

Options:
  -h, --help            show this help message and exit
  --dataDir=DATADIR     data dir [../zp_atmos_bias_summary]
  --atmos_params=ATMOS_PARAMS
                        bias atmospheric parameters
                        [airmass,ozone,aerosol,pwv]
  --obs=OBS             variable to use as ref [zp]
  --obs_unit=OBS_UNIT   unit variable to use as ref [mmag]
  --bands=BANDS         filters to plot [grizy]
  --outDir=OUTDIR       output directory[../iso_zp]

</pre>

[airmass bias](iso_zp/bias_airmass.png)

[ozone bias](iso_zp/bias_ozone.png)

[aerosol bias](iso_zp/bias_aerosol.png)

[pwv bias](iso_zp/bias_pwv.png)