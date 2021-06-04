from optparse import OptionParser
from sn_plotter_metrics.effiPlot import plotEffi
import numpy as np
from sn_plotter_metrics import plt


def plotEffiRate(data, healpixID):
    """
    Method to plot effi vs z and SN rate vs z

    Parameters
    ---------------
    data: pandas df
       data to process
    healpixID: int
      healpix id

    """

    idx = data['healpixID'] == healpixID

    sel = data[idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    seasons = np.unique(sel['season'])
    idx = seasons <= 10
    for season in seasons[idx]:

        pp.plotEffi_indiv(sel, ax, healpixID, season, 'effi', 'effi_err',
                          'Observing Efficiencies', ls='-', label='season {}'.format(season))

    ax.set_xlabel('z', weight='bold')
    ax.set_ylabel('Observing Efficiency', weight='bold')
    ax.set_xlim([0.01, 0.7])
    ax.set_ylim([0.0, None])
    ax.grid()

    rate, raterr = pp.getRates()
    zvals = np.arange(0.01, 0.7, 0.01)

    axb = ax.twinx()
    axb.plot(zvals, rate(zvals), color='k', ls='dashed')
    axb.set_ylabel('SN Rate', weight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.15), ncol=5, frameon=False, fontsize=14)


def plotCumul(data, healpixID):
    """
    Method to NSN cumul vs z

    Parameters
    ---------------
    data: pandas df
       data to process
    healpixID: int
      healpix id

    """

    idx = data['healpixID'] == healpixID

    sel = data[idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    seasons = np.unique(sel['season'])
    idx = seasons <= 10

    zvals = np.arange(0.01, 0.71, 0.01)

    for season in seasons[idx]:

        pp.plotCumul(sel, ax, healpixID, season,
                     label='season {}'.format(season))

    ax.set_xlabel('z', weight='bold')
    ax.set_ylabel('Normalised Cumulative $N_{SN}$', weight='bold')
    ax.set_xlim([0.01, 0.7])
    ax.set_ylim([0.0, None])
    ax.grid()
    ax.plot(zvals, [0.95]*len(zvals), color='k', ls='dashed')
    ax.text(0.2, 0.96, '95th percentile', fontsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.15), ncol=5, frameon=False, fontsize=14)


def plotNSN(data, healpixID):
    """
    Method to NSN cumul vs z

    Parameters
    ---------------
    data: pandas df
       data to process
    healpixID: int
      healpix id

    """

    idx = data['healpixID'] == healpixID

    sel = data[idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    seasons = np.unique(sel['season'])
    idx = seasons <= 10

    zvals = np.arange(0.01, 0.71, 0.01)

    for season in seasons[idx]:

        pp.plotNSN(sel, ax, healpixID, season,
                   label='season {}'.format(season))

    ax.set_xlabel('z', weight='bold')
    ax.set_ylabel('$N_{SN} (z <)$ ', weight='bold')
    ax.set_xlim([0.01, 0.7])
    ax.set_ylim([0.0, None])
    ax.grid()

    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.15), ncol=5, frameon=False, fontsize=14)


parser = OptionParser()

parser.add_option(
    '--metricDir', help='metric directory [%default]', default='MetricOutput', type=str)
parser.add_option(
    '--dbName', help='OS to process [%default]', default='daily_ddf_v1.5_10yrs', type=str)
parser.add_option(
    '--fieldName', help='DD field to consider [%default]', default='CDFS', type=str)


opts, args = parser.parse_args()

metricDir = opts.metricDir
dbName = opts.dbName
fieldName = opts.fieldName

pp = plotEffi(metricDir, dbName, fieldName)

data = pp.data

print(np.unique(data['healpixID']))

healpixID = 144428

plotEffiRate(data, healpixID)

plotCumul(data, healpixID)

plotNSN(data, healpixID)
# pp.plotEffi()

plt.show()
