import numpy as np
import sn_plotter_metrics.nsnPlot as nsn_plot
from sn_tools.sn_io import loopStack
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
import glob
import healpy as hp

def loadData(dirFile, dbName, metricName,fieldType,nside):

    search_path = '{}/{}/{}/*NSNMetric_{}*_nside_{}_*.hdf5'.format(dirFile, dbName, metricName, fieldType, nside)
    print('looking for', search_path)
    fileNames = glob.glob(search_path)

    metricValues = np.array(loopStack(fileNames, 'astropyTable'))

    return metricValues
    
def plotMollview(data, varName, leg, op, xmin, xmax,nside):
        """
        Method to display results as a Mollweid map

        Parameters
        ---------------
        data: pandas df
          data to consider
        varName: str
          name of the variable to display
        leg: str
          legend of the plot
        op: operator
          operator to apply to the pixelize data(median, sum, ...)
        xmin: float
          min value for the display
        xmax: float
         max value for the display

        """
        npix = hp.nside2npix(nside)

        hpxmap = np.zeros(npix, dtype=np.float)
        hpxmap = np.full(hpxmap.shape, 0.)
        hpxmap[data['healpixID'].astype(
            int)] += data[varName]

        norm = plt.cm.colors.Normalize(xmin, xmax)
        cmap = plt.cm.jet
        cmap.set_under('w')
        resleg = op(data[varName])
        if 'nsn' in varName:
            resleg = int(resleg)
        else:
            resleg = np.round(resleg, 2)
        title = '{}: {}'.format(leg, resleg)

        hp.mollview(hpxmap, min=xmin, max=xmax, cmap=cmap,
                    title=title, nest=True, norm=norm)
        hp.graticule()

        # save plot here
        name = leg.replace(' - ', '_')
        name = name.replace(' ', '_')
        
def jackknife(x, func):
    """Jackknife estimate of the estimator func"""
    n = len(x)
    idx = np.arange(n)
    return np.sum(func(x[idx!=i]) for i in range(n))/float(n)

def jackknife_var(x, func):
    """Jackknife estiamte of the variance of the estimator func."""
    n = len(x)
    idx = np.arange(n)
    j_est = jackknife(x, func)
    return (n-1)/(n + 0.0) * np.sum((func(x[idx!=i]) - j_est)**2.0
                                    for i in range(n))

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


metricValues = loadData(dirFile,dbName,metricName,fieldType,nside)

print(metricValues.dtype)
sntype = 'faint'
idx = metricValues['status_{}'.format(sntype)] == 1
idx &= metricValues['zlim_{}'.format(sntype)] > 0.
idx &= metricValues['nsn_zlim_{}'.format(sntype)] > 0.
data = pd.DataFrame(metricValues[idx])

seasons = data['season'].unique()
io = seasons <= 10
seasons = seasons[io]
fig, ax = plt.subplots()

"""
means = data.groupby(['season']).max().reset_index()

ax.plot(means['season'],means['zlim_faint'],'ko')
plt.show()
"""

for season in seasons:
    idx = data['season'] == season
    sel = data[idx]
    leg = 'zlim - season {}'.format(int(season))
    #plotMollview(sel,'zlim_faint',leg,np.median,0.1,0.4,nside)
    #ax.hist(sel['zlim_faint'],histtype='step',bins=20)
    ax.plot(sel['zlim_faint'],np.cumsum(sel['zlim_faint']))
    #print(jackknife(sel['zlim_faint'],np.mean))
    
    
plt.show()


print(test)

dbInfo = pd.DataFrame([[dirFile,dbName, 'test', 'test', 'test', 'r', '*']],
                      columns=['dirFile','dbName', 'newName', 'group', 'plotName', 'color', 'marker'])
dbInfo = pd.DataFrame([[dirFile,dbName, 'test', 'test', 'test', 'r', '*','fbs','1.7.1','rolling']],
                      columns=['dirFile','dbName', 'newName', 'group', 'plotName', 'color', 'marker','simuType','simuNum','family'])
metricdata = nsn_plot.NSNAnalysis(dbInfo.loc[0], metricName, fieldType,
                                  nside)

# Mollweid plots
dbName = dbName.split('_10yrs')[0]
metricdata.Mollview_median('zlim', '{} - zlim'.format(dbName))
#metricdata.Mollview_median('ebvofMW', 'E(B-V)')
metricdata.Mollview_median('cadence', '{} -  cadence'.format(dbName))
metricdata.Mollview_median('season_length', '{} -  season length'.format(dbName))
metricdata.Mollview_sum('nsn_med_faint', '{} -  NSN'.format(dbName))
for b in 'ugrizy':
    metricdata.Mollview_median('N_{}'.format(b), '{} - Nvisits - {} band'.format(dbName,b))

plt.show()

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
