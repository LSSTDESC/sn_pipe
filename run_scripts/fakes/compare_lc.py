import numpy as np
import h5py
from optparse import OptionParser
from sn_tools.sn_io import loopStack
from astropy.table import Table
import matplotlib.pyplot as plt


def simuName(simulator, x1, color, cutoff, ebv):
    """
    function to get the Simu file name

    Parameters
    ---------------
    simulator: str
      simulator name
    x1: float
      SN x1
    color: float
      SN color
    cutoff: str
      cuutoff value
    ebv: float
      E(B-V)

    Returns
    -----------
    simu file name

    """

    res = 'Simu_{}_Fake_{}_{}_{}_ebvofMW_{}_0'.format(
        simulator, x1, color, cutoff, ebv)

    return res


def getLC(fName, index):
    lc = Table.read(fName, path='lc_{}'.format(index))

    return lc


def getSimu(simu, x1, color, daymax, z):

    idx = np.abs(simu['x1']-x1) < 1.e-5
    idx &= np.abs(simu['color']-color) < 1.e-5
    idx &= np.abs(simu['daymax']-daymax) < 1.e-5
    idx &= np.abs(simu['z']-z) < 1.e-5

    return simu[idx]


def plotLCs(lc_full, lc_fast):

    bands = 'ugrizy'
    band_id = dict(
        zip(bands, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]))
    colors = dict(zip(bands, 'bcgyrm'))
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(10, 9))
    fig.suptitle(
        'z={}, E(B-V)={}'.format(np.round(lc_full.meta['z'], 2), lc_full.meta['ebvofMW']))
    plotLC(lc_full, ax, band_id, 'o', 'k', 'None')
    plotLC(lc_fast, ax, band_id, '*', 'r', 'r')

    plt.show()


def plotLC(table, ax, band_id, marker, color, mfc, inum=0):
    fontsize = 10
    plt.yticks(size=fontsize)
    plt.xticks(size=fontsize)
    for band in 'ugrizy':
        i = band_id[band][0]
        j = band_id[band][1]
        # ax[i,j].set_yscale("log")
        idx = table['band'] == 'LSST::'+band
        idx &= table['snr_m5'] >= 1.
        sel = table[idx]
        if len(sel) > 0:
            print('hello', band, inum, sel['phase', 'time'], i, j)
            # ax[band_id[band][0]][band_id[band][1]].errorbar(sel['time'],sel['mag'],yerr = sel['magerr'],color=colors[band])
            ax[i, j].plot(sel['time'], sel['fluxerr'],
                          marker=marker, color=color, lineStyle='None', mfc=mfc)
        if i > 1:
            ax[i, j].set_xlabel('MJD [day]', {'fontsize': fontsize})
        ax[i, j].set_ylabel('Flux [pe/sec]', {'fontsize': fontsize})
        ax[i, j].text(0.1, 0.9, band, horizontalalignment='center',
                      verticalalignment='center', transform=ax[i, j].transAxes)


def plotLC_orig(table, ax, band_id, colors, inum=0):
    fontsize = 10
    plt.yticks(size=fontsize)
    plt.xticks(size=fontsize)
    for band in 'ugrizy':
        i = band_id[band][0]
        j = band_id[band][1]
        # ax[i,j].set_yscale("log")
        idx = table['band'] == 'LSST::'+band
        sel = table[idx]
        print('hello', band, inum, len(sel), i, j)
        if len(sel) > 0:
            # ax[band_id[band][0]][band_id[band][1]].errorbar(sel['time'],sel['mag'],yerr = sel['magerr'],color=colors[band])
            ax[i, j].errorbar(sel['time'], sel['flux_e_sec'], yerr=sel['flux_e_sec']/sel['snr_m5'],
                              markersize=200000., color=colors[band], linewidth=1)
        if i > 1:
            ax[i, j].set_xlabel('MJD [day]', {'fontsize': fontsize})
        ax[i, j].set_ylabel('Flux [pe/sec]', {'fontsize': fontsize})
        ax[i, j].text(0.1, 0.9, band, horizontalalignment='center',
                      verticalalignment='center', transform=ax[i, j].transAxes)


parser = OptionParser()
parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1 [%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color [%default]")
parser.add_option("--ebv", type=float, default=0.0,
                  help="ebvofMW value[%default]")
parser.add_option("--bluecutoff", type=float, default=380.,
                  help="blue cutoff for SN spectrum[%default]")
parser.add_option("--redcutoff", type=float, default=800.,
                  help="red cutoff for SN spectrum[%default]")
parser.add_option("--error_model", type=int, default=1,
                  help="error model to consider [%default]")
parser.add_option("--prefixDir", type=str, default='Output_Simu',
                  help=" prefix dir name where files are located[%default]")
"""
parser.add_option("--simus", type=str, default='sn_fast,sn_cosmo',
                  help=" simulator to use[%default]")
parser.add_option("--snrmin", type=float, default=1.,
                  help="SNR min for LC points (fit)[%default]")
parser.add_option("--nbef", type=int, default=4,
                  help="min n LC points before max (fit)[%default]")
parser.add_option("--naft", type=int, default=10,
                  help="min n LC points after max (fit)[%default]")
parser.add_option("--nbands", type=int, default=0,
                  help="min number of bands with at least 2 points with SNR>5[%default]")

parser.add_option("--sigma_mu", type=int, default=0,
                  help="to estimate sigma mu[%default]")
"""
opts, args = parser.parse_args()

cutoff = 'error_model'
if opts.error_model == 0:
    cutoff = '{}_{}'.format(opts.bluecutoff, opts.redcutoff)

# main directory name
dirName = '{}_{}_ebvofMW_{}'.format(opts.prefixDir, cutoff, opts.ebv)

# get simu summary files
simu_fast_name = simuName('sn_fast', opts.x1, opts.color, cutoff, opts.ebv)
simu_full_name = simuName('sn_cosmo', opts.x1, opts.color, cutoff, opts.ebv)

lc_fast_name = simu_fast_name.replace('Simu', 'LC')
lc_full_name = simu_full_name.replace('Simu', 'LC')

simu_fast_fullname = '{}/{}.hdf5'.format(dirName, simu_fast_name)
simu_full_fullname = '{}/{}.hdf5'.format(dirName, simu_full_name)

lc_fast_fullname = '{}/{}.hdf5'.format(dirName, lc_fast_name)
lc_full_fullname = '{}/{}.hdf5'.format(dirName, lc_full_name)


simu_fast = loopStack([simu_fast_fullname], 'astropyTable')
simu_full = loopStack([simu_full_fullname], 'astropyTable')

simu_fast.convert_bytestring_to_unicode()
simu_full.convert_bytestring_to_unicode()

# loop
print(simu_full.columns)
for vv in simu_full:
    print('bbbbbb', vv['x1', 'color', 'daymax', 'z'])
    if vv['z'] >= 0.7:
        # get lc full here
        lc_full = getLC(lc_full_fullname, vv['index_hdf5'])
        simu_fastlc = getSimu(
            simu_fast, vv['x1'], vv['color'], vv['daymax'], vv['z'])
        lc_fast = getLC(lc_fast_fullname,
                        simu_fastlc['index_hdf5'].data.item())
        print(lc_full.meta)
        print(lc_fast.meta)
        plotLCs(lc_full, lc_fast)
    # break
