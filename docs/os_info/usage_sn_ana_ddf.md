## script plot_scripts/sn/sn_analyzer_sel_ddf.py

## Usage: sn_analyzer_sel_ddf.py [options]

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
  --ud_fields=UD_FIELDS
                        UD fields to consider [COSMOS,XMM-LSS]
  --dd_fields=DD_FIELDS
                        DD fields to consider [CDFS,ELAISS1,EDFS_a,EDFS_b]

</pre>

An example of the config input file is given [here](config_ana_selplot.csv)
