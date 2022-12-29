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

    print('alors leg',leg)
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
    print(selfi[['dist', 'healpixID']][:5])

    ax.plot(selfi[whatx], selfi[whaty], color=color, marker=marker, mfc=mfc)
    ax.set_xlabel(legx)
    ax.set_ylabel(legy, color=color)
    ax.tick_params(axis='y', colors=color)
    ax.grid()


def getVal(selfi, whatx='dist', whaty='nsn', frac=0.9):

    nmax = np.max(selfi[whaty])
    idx = selfi[whaty] <= frac*nmax

    return np.min(selfi[idx][whatx])


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

    ntot = np.sum(selfi['nsn'])
    bins = np.arange(0.1, 2.5, 0.1)
    r = []
    for bb in bins:
        bbi = [0., bb]
        group = selfi.groupby(pd.cut(selfi['dist'], bbi))
        for group_name, df_group in group:
            r.append(np.sum(df_group['nsn'])/ntot)

    print(r)

    # plot_centers = (bins[:-1] + bins[1:])/2

    ax.plot(bins, r, color=color, marker=marker, mfc=mfc)
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


def check_pixel(data, hpix):
    """
    Function to check if hpix is in data

    Parameters
    ---------------
    data: pandas df
      data to process
    hpix: int
      healpix id

    """

    idx = data['healpixID'] == hpix

    print('result for', hpix, data[idx])


def plot_Molls(df):

    print('plotting Moll',len(df))
    plot_Moll_season(nside, df)


parser = OptionParser()


parser.add_option("--field", type=str, default='COSMOS',
                  help="field [%default]")
parser.add_option("--inputFile", type=str, default='Summary_DD_pixel.hdf5',
                  help="input file name [%default]")
parser.add_option("--nside", type=int, default=128,
                  help="healPix nside parameter [%default]")
parser.add_option("--timeCol", type=str, default='season',
                  help="colname for time display[%default]")

opts, args = parser.parse_args()

fi = opts.inputFile
nside = opts.nside
field = opts.field.split(',')
timeCol = opts.timeCol

# loading data - os pixels
df = pd.read_hdf(fi)

print(df.columns)
print(df['dbName'].unique())
dbName = df['dbName'].unique()
pixArea = hp.nside2pixarea(nside, degrees=True)
#fig, ax = plt.subplots(figsize=(12, 8))
#figb, axb = plt.subplots(figsize=(12, 8))
colors = ['b', 'r']
for io, db in enumerate(dbName):
    idx = df['dbName'] == db
    idx &= df['field'] == field[io]
    print('allo1', len(df[idx]))
    sel = df[idx]
    for season in sel[timeCol].unique():
        idxb = sel[timeCol] == int(season)
        print('allo2', len(sel[idxb]))
        # plot Mollview here
        if season<=10:
            plot_Moll_season(nside,sel[idxb], leg='{} {} \n cadence [days]'.format(timeCol,season),maxval=10)
            plot_Hist(sel[idxb],figtitle='{} {}'.format(timeCol,season))
plt.show()
