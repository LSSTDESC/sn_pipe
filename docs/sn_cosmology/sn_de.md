# Dark Energy parameter plots

## Usage: plot_scripts/cosmology/plot_cosmo.py [options]

<pre>

Script to plot cosmology from SNe Ia

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location dir[../cosmo_fit_TiDES_notelrot_TiDES_5]
  --config=CONFIG       config for the plots [config_ana_selplot.csv]
  --timescale=TIMESCALE
                        timescale for plot - year or season [year]
  --UDFs=UDFS           UD fields [COSMOS,XMM-LSS]
  --DFs=DFS             Deep fields [CDFS,EDFS,ELAISS1,EDFS_a,EDFS_b]
  --comment_on_plot=COMMENT_ON_PLOT
                        comment for the SMoM plot [Host spectro-z only]
  --fill_between=FILL_BETWEEN
                        to fill +-1 sigma area with yellow [0]
  --prior=PRIOR         data were processed with or withuot prior [1]
  --plots=PLOTS         plots to make
                        [mom_year,mom_survey,sigma_w0,sigma_wa,nsn]
  --ref_OS=REF_OS       ref os to normalize the plots [None]
  --spectro_config=SPECTRO_CONFIG
                        spectro config [WFD_TiDES]
  --year_max=YEAR_MAX   year max for the display [6]

</pre>

## Example
python plot_scripts/cosmology/plot_cosmo.py --config=[config_ana_selplot.csv](config_ana_selplot.csv) --dbDir=../cosmo_fit_notelrot_TiDES_5

[SMoM vs year](plot1a.png)

[SMoM vs OS](plot2a.png)

[sigma_w0 vs year](plot3a.png)

[sigma_wa vs year](plot4a.png)

[NSN vs year - COSMOS](plot5a.png)

[NSN vs year - XMM-LSS](plot6a.png)

[NSN vs year - ELAISS1](plot7a.png)

[NSN vs year - CDFS](plot8a.png)

[NSN vs year - EDFS_a](plot9a.png)

[NSN vs year - EDFS_b](plot10a.png)

[NSN vs year - all DDFs](plot11a.png)
