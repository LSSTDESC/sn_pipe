import pandas as pd
import numpy as np
import healpy as hp
from sn_plotter_metrics import plt
from optparse import OptionParser
from sn_tools.sn_io import loopStack
import glob


def plotMollview(nside, data, varName, leg, op, xmin, xmax):
    """
    MethodFunction to display results as a Mollweid map

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

    idx = hpxmap > 0
    print(hpxmap[idx])
    norm = plt.cm.colors.Normalize(xmin, xmax)
    import copy
    cmap = copy.copy(plt.cm.jet)
    cmap.set_under('w')
    cmap.set_over('grey')
    """
    resleg = op(data[varName])
    if 'nsn' in varName:
        resleg = int(resleg)
    else:
        resleg = np.round(resleg, 2)
    # title = '{}: {}'.format(leg, resleg)
    # title = 'bbb'
    """
    hp.mollview(hpxmap, min=xmin, max=xmax, cmap=cmap,
                title=leg, nest=True, norm=norm)
    hp.graticule()


def plot_Moll_season(nside, data, what='cadence_mean', leg='cadence [days]', minval=1, maxval=20):
    """
    function to plot data using Mollview

    Parameters
    --------------
    nside: int
      healpix nside parameter
    data: pandas df
      data to plot
    what: str, opt
      col to plot (default: cadence_mean)
    leg: str, opt
      legend for the plot (default: cadence [days])
    minval: float, opt
      min value for the plot (default: 1)
    maxval: float, opt
      max value for the plot (default: 20)

    """

    plotMollview(nside, data, what, leg, np.median, minval, maxval)


def get_dist(data, pixRA_mean=-1, pixDec_mean=-1):
    """
    Function to estimate the distance dist = sqrt((deltaRA*cos(Dec))**2+deltaDec**2)

    Parameters
    ---------------
    data: pandas df
      data to process

    Returns
    ----------
    pandas df with dist col

    """
    if pixRA_mean == -1:
        pixRA_mean = np.mean(data['pixRA'])
        pixDec_mean = np.mean(data['pixDec'])
    data['dist'] = np.sqrt(((data['pixRA']-pixRA_mean)*np.cos(np.deg2rad(data['pixDec'])))**2
                           + (data['pixDec']-pixDec_mean)**2)
    data['pixRA_mean'] = pixRA_mean
    data['pixDec_mean'] = pixDec_mean

    return data


def plot_2D(selfi, whatx='dist', legx='radius [deg]', whaty='cadence_mean', legy='Mean cadence [days]', figtitle='', fig=None, ax=None, color='k', marker='o', mfc='k', sort=True):
    """
    Method to plot 2D variables

    Parameters
    --------------
    sel: pandas df
      data to plot
    field: str
      field to plot
    seas: int
      season of observation
    whatx: str, opt
      x var to plot (default:dist)
    legx: str, opt
      xlegend (default:radius [deg])
    whaty: str, opt
      y var to plot (default:cadence_mean)
    legy: str, opt
      ylegend (default:Mean cadence [days])
    figtitle: str, opt
      title for the figure (default: '')
    fig: figure, opt
      mpl figure (default: None)
    ax: axis, opt
      mpl axis (default: None)

    """

    # make the plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 9))

    if figtitle:
        fig.suptitle(figtitle)

    if sort:
        selfi = selfi.sort_values(by=['dist'])
    selfi = selfi.fillna(0.0)
    print(selfi[['dist', 'healpixID']])

    ax.plot(selfi[whatx], selfi[whaty], color=color, marker=marker, mfc=mfc)
    ax.set_xlabel(legx)
    ax.set_ylabel(legy, color=color)
    ax.tick_params(axis='y', colors=color)
    ax.grid()


def plot_cumsum(selfi, whatx='dist', legx='radius [deg]', figtitle='', fig=None, ax=None, color='k', marker='o', mfc='k', sort=True):
    """
    Method to plot 2D variables

    Parameters
    --------------
    sel: pandas df
      data to plot
    field: str
      field to plot
    seas: int
      season of observation
    whatx: str, opt
      x var to plot (default:dist)
    legx: str, opt
      xlegend (default:radius [deg])
    figtitle: str, opt
      title for the figure (default: '')
    fig: figure, opt
      mpl figure (default: None)
    ax: axis, opt
      mpl axis (default: None)

    """

    # make the plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 9))

    if figtitle:
        fig.suptitle(figtitle)

    if sort:
        selfi = selfi.sort_values(by=[whatx])
    selfi = selfi.fillna(0.0)
    print(selfi[['dist', 'healpixID']])

    bins = np.arange(0., 2.5, 0.01)
    group = selfi.groupby(pd.cut(selfi['dist'], bins))

    print('group', group)
    for group_name, df_group in group:
        print(group_name, df_group)

    plot_centers = (bins[:-1] + bins[1:])/2
    plot_values = np.sum(group['nsn'])/np.sum(selfi['nsn'])
    print(plot_values)
    print(test)
    cumsum = np.cumsum(selfi[whatx].to_list())
    print('lll', cumsum)
    ax.plot(selfi[whatx], cumsum/cumsum[-1],
            color=color, marker=marker, mfc=mfc)
    ax.set_xlabel(legx)
    ax.grid()


def plot_Hist(data, whatx='cadence_mean', bins=range(1, 20), density=True, legx='Mean cadence [days]', legy='Pixel fraction', figtitle='', fig=None, ax=None):
    """
    Function to plot histogram

    Parameters
    --------------
    data: pandas df
      data to plot
    whatx: str, opt
      col to plot (default: cadence_mean)
    bins: list, opt
     histo bins (default: range(1,20))
    density: bool, opt
     to norm the histo (default: True)
    figtitle: str, opt
      title for the figure (default: '')
    legx: str, opt
      x-axis legend (default:Mean cadence [days])
    legy: str, opt
      y-axis legend (default: Pixel fraction)
  fig: figure, opt
      mpl figure (default: None)
    ax: axis, opt
      mpl axis (default: None)

    """
    fig, ax = plt.subplots(figsize=(10, 9))
    # make the plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 9))

    fig.suptitle(figtitle)
    ax.hist(data[whatx], histtype='step',
            bins=bins, density=density, color='k', lw=2)
    ax.grid()
    ax.set_xlabel(legx)
    ax.set_ylabel(legy)


def load_metric(dirFile, dbName, metricName, fieldName, nside):

    search_path = '{}/{}/{}_{}/*{}Metric*nside_{}_*.hdf5'.format(
        dirFile, dbName, metricName,  fieldName, metricName, nside)
    fileNames = glob.glob(search_path)
    print('fileNames', search_path, fileNames)
    metricValues = pd.DataFrame()
    if fileNames:
        metricValues = loopStack(fileNames, 'astropyTable').to_pandas()

    return metricValues


parser = OptionParser()

parser.add_option("--dbName", type=str, default='alt_sched',
                  help="db name [%default]")
parser.add_option("--field", type=str, default='COSMOS',
                  help="field [%default]")
parser.add_option("--season", type=str, default="1",
                  help="season of observation [%default]")
parser.add_option("--inputFile", type=str, default='Summary_DD_pixel.hdf5',
                  help="season of observation [%default]")
parser.add_option("--nside", type=int, default=128,
                  help="healPix nside parameter [%default]")
parser.add_option("--dbDir", type=str, default='../MetricOutput_DD_new_128_gnomonic_circular',
                  help="main location dir of metric files [%default]")
parser.add_option("--metric", type=str, default='NSNY',
                  help="metric name [%default]")

opts, args = parser.parse_args()

fi = opts.inputFile
nside = opts.nside
dbName = opts.dbName.split(',')
field = opts.field.split(',')
season = opts.season.split(',')
dbDir = opts.dbDir
metric = opts.metric

# loading data - os pixels
df = pd.read_hdf(fi)

print(df.columns)
for io, db in enumerate(dbName):
    # loading metric results
    metricValues = load_metric(dbDir, db, metric, field[io], nside)
    idx = metricValues['zcomp'] > 0.
    metricValues = metricValues[idx]
    idx = df['dbName'] == db
    idx &= df['season'] == int(season[io])
    idxb = metricValues['season'] == int(season[io])
    # plot Mollview here
    plot_Moll_season(nside, df[idx])
    plot_Moll_season(nside, metricValues[idxb], what='nsn')
    idx &= df['field'] == field[io]
    suptit = '{} - season {}'.format(field[io], int(season[io]))
    suptit += '\n {}'.format(db)
    fig, ax = plt.subplots(figsize=(12, 8))
    # get distance
    selfi = get_dist(df[idx])
    plot_2D(selfi, figtitle=suptit, fig=fig, ax=ax, mfc='None')
    plot_cumsum(selfi, mfc='None')
    mmet = get_dist(metricValues[idxb], pixRA_mean=np.mean(
        selfi['pixRA_mean']), pixDec_mean=np.mean(selfi['pixDec_mean']))
    plot_cumsum(mmet, whatx='zcomp', legx='$z_{complete}$', mfc='None')
    plot_cumsum(mmet, whatx='nsn', legx='$N_{SN}$', mfc='None')
    """
    plot_2D(mmet, whaty='nsn', legy='N$_{SN}$',
            figtitle=suptit, fig=fig, ax=ax.twinx(), color='b', marker='s', mfc='None')
    """
    plot_2D(mmet, whaty='zcomp', legy='$z_{complete}$',
            figtitle=suptit, fig=fig, ax=ax.twinx(), color='b', marker='s', mfc='None')

    print('median', mmet['zcomp'].median())
    """
    plot_Hist(df[idx], figtitle=suptit)
    print(np.unique(df[idx]['filter_alloc']))
    """
plt.show()
