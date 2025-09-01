# LSST SN survey analysis

## Analysis

### python run_scripts/sn_analysis/nsn_surveys.py

Usage: script to analyze LSST SN surveys

<pre>
Options:
  -h, --help         show this help message and exit
  --dataDir=DATADIR  data directory [../sn_surveys]
  --dbList=DBLIST    data directory [list_surveys.csv]
  --outDir=OUTDIR    output directory [../sn_summary_surveys]

</pre>

### Example
python run_scripts/sn_analysis/nsn_surveys.py  --dbList=[list_surveys.csv](list_surveys.csv)


## Results

### python plot_scripts/cosmology/plot_sn_survey.py

Usage: script to plot LSST SN surveys

<pre>
Options:
  -h, --help            show this help message and exit
  --dataDir=DATADIR     data directory [../sn_summary_surveys]
  --surveyList=SURVEYLIST
                        OS for DD [list_surveys_plot.csv]
  --plots=PLOTS         plots to draw [nsn_all,nsn_ud]
  --print_nsn=PRINT_NSN
                        to print nsn as a latex table [0]
  --ud_fields=UD_FIELDS
                        UD fields to consider [COSMOS,XMM-LSS]
  --dd_fields=DD_FIELDS
                        DD fields to consider [CDFS,ELAISS1,EDFS_a,EDFS_b]
  --config=CONFIG       OS DD list[input/plots/config_ana.csv]

</pre>

### Example

python plot_scripts/cosmology/plot_sn_survey.py --config=[config_ana_selplot.csv](config_ana_selplot.csv) --surveyList=[list_surveys_plot.csv](list_surveys_plot.csv)

Plots are similar to the ones [here](../sn/sn_summary.md)