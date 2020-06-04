import matplotlib.pyplot as plt
import h5py
import numpy as np
from astropy.table import Table, vstack
import pprint
from optparse import OptionParser


def plotParameters(fieldname, fieldid, tab, season):
    """ Plot simulation parameters
    parameters ('X1', 'Color', 'DayMax', 'z')
    Input
    ---------
    fieldname: (DD or WFD)
    fieldid: (as given by OpSim)
    tab: recarray of parameters
    season: season

    Returns
    ---------
    Plot (x1,color,dayMax,z)
    """

    idx = tab['season'] == season
    sel = tab[idx]
    thesize = 15
    toplot = ['x1', 'color', 'daymax', 'z']
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 9))
    title = '{} - fieldid {} - season {}'.format(fieldname, fieldid, season)
    fig.suptitle(title, fontsize=thesize)

    for i, var in enumerate(toplot):
        ix = int(i/2)
        iy = i % 2
        axis = ax[ix][iy]
        i  # f var != 'z':
        axis.hist(sel[var], histtype='step')  # bins=len(sel[var]))
        axis.set_xlabel(var, fontsize=20)
        axis.set_ylabel('Number of entries', fontsize=thesize)
        axis.tick_params(axis='x', labelsize=thesize)
        axis.tick_params(axis='y', labelsize=thesize)


def load_params(paramFile):
    """
    Function to load simulation parameters

    Parameters
    ---------------
    paramFile: str
       name of the parameter file

    Returns
    -----------
    params: astropy table
       with simulation parameters

    """

    f = h5py.File(paramFile, 'r')
    print(f.keys(), len(f.keys()))
    params = Table()
    for i, key in enumerate(f.keys()):
        pars = Table.read(paramFile, path=key)
        params = vstack([params, pars])

    return params


def plotLC(table, ax, band_id, inum=0):
    """
    Method to plot produced LC

    Parameters
    ---------------
    table: astropy table
       LC
    ax: matplotlib axis
      to display
    band_id: int
      id of the band
    inum: int, opt
      ?? (default: 0)
    """

    fontsize = 10
    plt.yticks(size=fontsize)
    plt.xticks(size=fontsize)
    for band in 'ugrizy':
        i = band_id[band][0]
        j = band_id[band][1]
        # ax[i,j].set_yscale("log")
        idx = table['band'] == 'LSST::'+band
        sel = table[idx]
        # print('hello',band,inum,len(sel))
        #ax[band_id[band][0]][band_id[band][1]].errorbar(sel['time'],sel['mag'],yerr = sel['magerr'],color=colors[band])
        ax[i, j].errorbar(sel['time'], sel['flux_e_sec'], yerr=sel['flux_e_sec']/sel['snr_m5'],
                          markersize=200000., color=colors[band], linewidth=1)
        if i > 1:
            ax[i, j].set_xlabel('MJD [day]', {'fontsize': fontsize})
        ax[i, j].set_ylabel('Flux [pe/sec]', {'fontsize': fontsize})
        ax[i, j].text(0.1, 0.9, band, horizontalalignment='center',
                      verticalalignment='center', transform=ax[i, j].transAxes)


parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="db name [%default]")
parser.add_option("--dbDir", type="str", default='Output_Simu',
                  help="dir location of the results [%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

parName = 'Simu_{}.hdf5'.format(opts.dbName)

params = load_params('{}/{}'.format(opts.dbDir, parName))

print(params)

lcFile = '{}/LC_{}.hdf5'.format(opts.dbDir, opts.dbName)
f = h5py.File(lcFile, 'r')
print(f.keys(), len(f.keys()))

bands = 'ugrizy'
band_id = dict(zip(bands, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]))
colors = dict(zip(bands, 'bcgyrm'))

zref = 0.7
for i, key in enumerate(f.keys()):

    lc = Table.read(lcFile, path=key)

    if np.abs(lc.meta['z']-zref) < 1.e-5:
        fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(12, 8))
        pprint.pprint(lc.meta)  # metadata
        figtitle = '($x_1,c$)=({},{})'.format(
            lc.meta['x1'], lc.meta['color'])
        figtitle += ' - z={}'.format(lc.meta['z'])
        figtitle += ' \n daymax={}'.format(lc.meta['daymax'])
        fig.suptitle(figtitle)

        # print(lc)  # light curve points
        plotLC(lc, ax, band_id, i)
        plt.draw()
        plt.pause(5)
        plt.close()
        # break
