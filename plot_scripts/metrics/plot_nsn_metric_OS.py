import numpy as np
import sn_plotter_metrics.nsnPlot as nsn_plot
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt

parser = OptionParser(
    description='Display NSN metric results for WFD fields')

parser.add_option("--dirFile", type="str", default='/sps/lsst/users/gris/MetricOutput',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='WFD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--dbName", type="str", default='descddf_v1.4_10yrs',
                  help="dbname to process [%default]")


opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
nside = opts.nside
fieldType = opts.fieldType
metricName = 'NSN'
dbName = opts.dbName


dbInfo = pd.DataFrame([[dbName, 'test', 'test', 'test', 'r', '*']],
                      columns=['dbName', 'newName', 'group', 'plotName', 'color', 'marker'])

metricdata = nsn_plot.NSNAnalysis(dirFile, dbInfo.loc[0], metricName, fieldType,
                                  nside)

# Mollweid plots
metricdata.Mollview_median('zlim', 'zlim')
metricdata.Mollview_median('ebvofMW', 'E(B-V)')
metricdata.Mollview_median('cadence', 'cadence')
metricdata.Mollview_median('season_length', 'season length')
metricdata.Mollview_sum('nsn_med', 'NSN')

# Correlation plots
# get the data to plot
data = metricdata.data

vars = [('cadence', 'cadence')]
# estimate effective cadence per band
for b in 'ugrizy':
    data['cadence_{}'.format(b)] = data['season_length']/data['Nvisits_{}'.format(
        b)]
    vars.append(('cadence_{}'.format(b), 'cadence - {}'.format(b)))

data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

var = ('nsn_med', '#of supernovae')

datamed = data.groupby(['healpixID']).median()
datasum = data.groupby(['healpixID']).sum()

for vv in vars:
    metricdata.plotCorrel(datamed, datasum, x=vv, y=var)


print('data names', metricdata.data.columns)
#metricdata.plot_season(metricdata.data, 'm5_med', np.median)
plt.show()
