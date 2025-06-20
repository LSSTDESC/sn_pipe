## Usage: plot_scripts/os_info/plot_ddf_visits_night.py [options]

<pre>
Script to analyse DDF visits on a nightly basis from pointings

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         file directory [../ddf_visits_night]
  --dbName=DBNAME       OS name [desc_ddf_v4.2.1_10yrs]
  --dbList=DBLIST       dbList to process [dbList.csv]
  --nproc=NPROC         number of procs for multiprocessing [8]
  --configplot=CONFIGPLOT
                        configuration for the plot [config_ana_selplot.csv]
  --DDF=DDF             DDF to consider [DD:COSMOS,DD:XMM_LSS]
  --plots=PLOTS         plots to make [plot_indiv,plot_all]
  
</pre>

##example

###python plot_scripts/os_info/plot_ddf_visits_night.py --plots=plot_indiv

[Filter sequences](plotb1.png)

[Filter sequences - with y band](plotb2.png)

[Filter sequences - with u band](plotb3.png)

[Filter sequences - with no u nor y obs.](plotb4.png)

###python plot_scripts/os_info/plot_ddf_visits_night.py --plots=plot_night

[DDF obs. time and NDDF observed vs night](plotb5.png)

###python plot_scripts/os_info/plot_ddf_visits_night.py --plots=plot_ana

[Main filter sequences vs year - COSMOS](plotb6.png)

[Fraction of nights with n_obs/nexp=1 vs year - COSMOS](plotb7.png)

[Fraction of nights with n_obs/nexp>1 vs year - COSMOS](plotb8.png)

[Fraction of nights with n_obs/nexp<1 vs year - COSMOS](plotb9.png)

[Fraction of nights vs n_obs/n_exp for desc_ddf_gen_0.80_sn_v4.3.1_10yrs, XMM_LSS, year=4, n_obs/n_exp>1](plotb10.png)

[Fraction of nights vs n_obs/n_exp for desc_ddf_gen_0.80_sn_v4.3.1_10yrs, XMM_LSS, year=4, n_obs/n_exp<1](plotb11.png)