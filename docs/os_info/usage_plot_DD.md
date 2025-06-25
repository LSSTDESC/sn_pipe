## Usage:  plot_scripts/metrics/plot_DD_stat.py

<pre>
OS analysis plots from pointings

Options:
  -h, --help            show this help message and exit
  --dirFile=DIRFILE     file directory
                        [../MetricOutput_DD_new_128_gnomonic_circular]
  --nside=NSIDE         nside for healpixels [128]
  --fieldType=FIELDTYPE
                        field type - DD, WFD, Fake [DD]
  --fieldNames=FIELDNAMES
                        fields to process [COSMOS,CDFS,                  XMM-
                        LSS,ELAISS1,EDFSa,EDFSb,EDFS]
  --metric=METRIC       metric name [NSNY]
  --pointingFile=POINTINGFILE
                        pointing file name [Summary_DD_pointings.hdf5]
  --config=CONFIG       pointing file name [DD_fbs_2.99_plot.csv]
  --addMetric=ADDMETRIC
                        to add metric correlation plots [0]
  --dbName_night=DBNAME_NIGHT
                        dbName for night plot stat [baseline_v3.0_10yrs]
  --fieldName_night=FIELDNAME_NIGHT
                        field for night plot stat [COSMOS]
  --plots=PLOTS         plots to draw [summary,field_cad_seasonlength,field_nv
                        isits,field_nvisits_band,relative_depth]
</pre>

## Example

python plot_scripts/metrics/plot_DD_stat.py --dirFile . --config=config_ana_selplot.csv

A config file example is available [here](config_ana_selplot.csv).

Plots produced:

[DD budget](plot1.png)

[DD budget per field](plot3.png)

[Relative DD budget per field](plot2.png)

[Overhead (filter changes)](plot6.png)

[Overhead (filter changes) per field](plot4.png)

[Number of filter changes](plot7.png)

[Number of filter changes per field](plot5.png)

[Cadence and season length - COSMOS](plot8.png)

[Cadence and season length - XMM-LSS](plot9.png)

[Cadence and season length - CDFS](plot10.png)

[Cadence and season length - ELAISS1](plot13.png)

[Cadence and season length - EDFSa](plot11.png)

[Cadence and season length - EDFSb](plot12.png)

[Filter allocation per observing night - desc_ddf_v4.2.1 - COSMOS - season 1](plot14.png)