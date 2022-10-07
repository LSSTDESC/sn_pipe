import pandas as pd
import numpy as np
import healpy as hp
from sn_plotter_metrics import plt
from optparse import OptionParser


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


def get_dist(data):
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
    pixRA_mean = np.mean(data['pixRA'])
    pixDec_mean = np.mean(data['pixDec'])
    data['dist'] = np.sqrt(((data['pixRA']-pixRA_mean)*np.cos(np.deg2rad(data['pixDec'])))**2
                           + (data['pixDec']-pixDec_mean)**2)
    return data


def plot_2D(sel, whatx='dist', legx='radius [deg]', whaty='cadence_mean', legy='Mean cadence [days]', figtitle='', fig=None, ax=None):
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

  # get distance
    selfi = get_dist(sel)

    # make the plot
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 9))

    if figtitle:
        fig.suptitle(figtitle)

    ax.plot(selfi[whatx], selfi[whaty], 'ko')
    ax.set_xlabel(legx)
    ax.set_ylabel(legy)
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

opts, args = parser.parse_args()

fi = opts.inputFile
nside = opts.nside
dbName = opts.dbName.split(',')
field = opts.field.split(',')
season = opts.season.split(',')

# loading data
df = pd.read_hdf(fi)


for io, db in enumerate(dbName):
    idx = df['dbName'] == db
    idx &= df['season'] == int(season[io])
    # plot Mollview here
    plot_Moll_season(nside, df[idx])
    idx &= df['field'] == field[io]
    suptit = '{} - season {}'.format(field[io], int(season[io]))
    suptit += '\n {}'.format(db)
    plot_2D(df[idx], figtitle=suptit)
    plot_Hist(df[idx], figtitle=suptit)


plt.show()
