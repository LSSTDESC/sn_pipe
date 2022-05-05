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
    thecol = 'm'
    lw = 3
    idx = data['healpixID'] == healpixID

    sel = data[idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(top=0.85)
    seasons = np.unique(sel['season'])
    idx = seasons <= 10
    lls = ['solid', 'dotted', 'dashed']*3
    lls.append('dashdot')
    mm = ['o', 'v', 's']
    mm += ['s', 'o', 'v']
    mm += ['v', 's', 'v']
    mm.append('s')
    lsb = dict(zip(range(10), lls))
    mmb = dict(zip(range(10), mm))
    # for season in seasons[idx]:
    for season in [1, 6, 10]:
        pp.plotEffi_indiv(sel, ax, healpixID, season, 'effi', 'effi_err',
                          'Observing Efficiencies', ls=lsb[season-1], label='season {}'.format(season), marker=mmb[season-1], lw=lw)

    ax.set_xlabel('z')
    ax.set_ylabel('Observing Efficiency')
    ax.set_xlim([0.01, 0.7])
    ax.set_ylim([0.0, None])
    ax.grid()

    rate, raterr = pp.getRates()
    zvals = np.arange(0.01, 0.7, 0.01)

    axb = ax.twinx()
    axb.plot(zvals, rate(zvals), color=thecol, ls='dashed', lw=lw)
    axb.set_ylabel('SN Ia Rate', color=thecol)
    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.20), ncol=4, frameon=False)
    axb.tick_params(axis='y', colors=thecol)
    axb.spines['right'].set_color(thecol)


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
    lw = 3
    lls = ['solid', 'dotted', 'dashed']*3
    lls.append('dashdot')
    mm = ['o', 'v', 's']
    mm += ['s', 'o', 'v']
    mm += ['v', 's', 'v']
    mm.append('s')
    lsb = dict(zip(range(10), lls))
    mmb = dict(zip(range(10), mm))

    idx = data['healpixID'] == healpixID

    sel = data[idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(top=0.85)
    seasons = np.unique(sel['season'])
    idx = seasons <= 10

    zvals = np.arange(0.01, 0.71, 0.01)

    seasons = [1, 6, 10]
    shiftplot = dict(zip(seasons, [0., 0.1, 0.2]))
    # for season in seasons[idx]:
    for i, season in enumerate(seasons):

        pp.plotCumul(sel, ax, healpixID, season, shiftplot=shiftplot[season],
                     label='season {}'.format(season), ls=lsb[season-1], marker=mm[season-1], lw=lw)

    ax.set_xlabel('z')
    ax.set_ylabel('Normalised Cumulative $\mathrm{N_{SN}}$')
    ax.set_xlim([0.01, 0.7])
    ax.set_ylim([0.0, 1.])
    ax.grid()
    ax.plot(zvals, [0.95]*len(zvals), color='k', ls='dashed')
    ax.text(0.2, 0.96, '95th percentile')
    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.20), ncol=4, frameon=False)


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
    lls = ['solid', 'dotted', 'dashed']*3
    lls.append('dashdot')
    mm = ['o', 'v', 's']
    mm += ['s', 'o', 'v']
    mm += ['v', 's', 'v']
    mm.append('o')
    lsb = dict(zip(range(10), lls))
    mmb = dict(zip(range(10), mm))
    idx = data['healpixID'] == healpixID

    sel = data[idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(top=0.85)
    seasons = np.unique(sel['season'])
    idx = seasons <= 10

    zvals = np.arange(0.01, 0.71, 0.01)
    seasons = [1, 6, 10]
    shiftplot = dict(zip(seasons, [0., 2., 4.]))
    # for season in seasons[idx]:
    for season in seasons:
        pp.plotNSN(sel, ax, healpixID, season, shiftplot=shiftplot[season],
                   label='season {}'.format(season), ls=lsb[season-1], marker=mm[season-1])

    ax.set_xlabel('z')
    ax.set_ylabel('$\mathrm{N_{SN} (z <)}$ ')
    ax.set_xlim([0.01, 0.7])
    ax.set_ylim([0.0, None])
    ax.grid()

    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.20), ncol=4, frameon=False)

    """
    cell_text = []
    rob = []
    columns = []
    for key, val in ro.items():
        rob.append(val)
        columns.append(key)
    cell_text.append(rob)

    rows = ['NSN']

    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          # rowColours=colors,
                          colLabels=columns,
                          loc='center')
    """


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

#plotNSN(data, healpixID)
# pp.plotEffi()

plt.show()
