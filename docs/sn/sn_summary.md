# Usage: plot_scripts/sn/sn_analyzer_sel_ddf.py [options]

<pre>

Script to analyze SN - DDF after selection

Options:
  -h, --help            show this help message and exit
  --dbDir=DBDIR         OS location
                        dir[../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA]
  --config=CONFIG       OS DD list[input/plots/config_ana.csv]
  --norm_factor=NORM_FACTOR
                        normalization factor [30]
  --budget_DD=BUDGET_DD
                        DD budget [0.07]
  --runType=RUNTYPE     run type  [spectroz]
  --timescale=TIMESCALE
                        timescale of the files to process [year]
  --timeslots=TIMESLOTS
                        time slot (season or year) to process [1-10]
  --dataType=DATATYPE   data type [DataFrame]
  --plots=PLOTS         plots to draw [nsn_all,nsn_ud]

</pre>

# Example

python plot_scripts/sn/sn_analyzer_sel_ddf.py --dbDir=../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA --config=config_ana_selplot.csv

[sum(NSN) vs year - all DDFs](plota1.png)

[cumulative sum(NSN) vs year - all DDFs](plota2.png)

[sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - all DDFs](plota3.png)

[cumulative sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - all DDFs](plota4.png)

[Ratio  NSN(z>=0.8, sigmaC<=0.04)/NSN(z>=0.8) vs year - all DDFs](plota5.png)


[sum(NSN) vs year - COSMOS+XMM-LSS](plota6.png)

[cumulative sum(NSN) vs year - COSMOS+XMM-LSS](plota7.png)

[sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - COSMOS+XMM-LSS](plota8.png)

[cumulative sum(NSN)(z>=0.8, sigmaC<=0.04) vs year - COSMOS+XMM-LSS](plota9.png)

[Ratio  NSN(z>=0.8, sigmaC<=0.04)/NSN(z>=0.8) vs year - COSMOS+XMM-LSS](plota10.png)