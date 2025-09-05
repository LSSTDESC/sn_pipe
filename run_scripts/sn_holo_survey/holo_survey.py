#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:17:54 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import numpy as np
# from sn_tools.sn_fp_pixel import get_pixels_in_window, FocalPlane, get_xy_pixels
from sn_tools.sn_fp_pixel import Pixels_in_FP
import matplotlib.pyplot as plt
import matplotlib
import healpy as hp
from astropy.time import Time
from sn_scheduler.scheduler import StarAltTime
import astropy.units as u
import pandas as pd
from sn_tools.sn_obs import getPix

plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['grid.alpha'] = 0.75  # transparency, between 0.0 and 1.0
plt.rcParams['grid.linestyle'] = '-'  # simple line
plt.rcParams['grid.linewidth'] = 0.4  # in points
plt.rcParams['font.size'] = 13
plt.rcParams['font.weight'] = "bold"


def plotMollview(pixels, axa, fig, night, nside, comment='', n=6):
    """
    Method to display a Mollweid view

    Parameters
    --------------
    pixels: pandas df
      data to plots
    axa: matplotlib axis
      axis to use for the plot
    fig: matplotlib figure
      figure to use for plot
    night: int
      night number
    """

    xmin = 0.99999
    xmax = np.max([np.max(pixels['color']), 1])

    norm = plt.cm.colors.Normalize(xmin, xmax)
    # cmap = plt.get_cmap('jet', int(xmax))
    # n = int(xmax)+1
    # n = 6
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cmap = from_list(None, plt.cm.Set1(range(1, n)), n-1)
    cmap.set_under('w')

    npixels = hp.nside2npix(nside)
    hpxmap = np.zeros(npixels, dtype=int)
    hpxmap = np.full(hpxmap.shape, -2)
    hpxmap[pixels['healpixID']] = pixels['color'].astype(int)

    # print('hello ',xmin,xmax)
    plt.axes(axa)
    hp.mollview(hpxmap, nest=True, cmap=cmap,
                min=xmin, max=n, norm=norm, cbar=False,
                title=comment, hold=True, badcolor='white', xsize=800)

    hp.graticule(verbose=False)

    """
    ax = plt.gca()
    image = ax.get_images()[0]
    cbar = fig.colorbar(image, ax=ax, ticks=range(
        1, n), orientation='horizontal')
    # cbar = fig.colorbar(ax[0,0], ticks=range(0,n), orientation='horizontal')  # set some values to ticks

    labels = list(range(1, n))

    tick_label = list(map(str, labels))

    tick_label[-1] = '>{}'.format(tick_label[-2])
    # print(tick_label)
    cbar.ax.set_xticklabels([])
    cbar.ax.tick_params(size=0)
    for j, lab in enumerate(tick_label):
        cbar.ax.text(labels[j]+0.5, -10., lab)

    # ax.text(-3.5, 0.9, self.dbName, fontsize=15, color='r')
    ax.text(-3.5, 0.6, 'night {}'.format(night), fontsize=15, color='k')
    """


def process_night(stars_alt, mjd, targets, fig, ax):
    """


    Parameters
    ----------
    stars_alt : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.
    month : TYPE
        DESCRIPTION.
    day : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    plot_it : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    targets_info : TYPE
        DESCRIPTION.

    """
    """
    Function to process targets

    Parameters
    ----------
    stars_alt : StarAltTime instance
        The class where the calculation is made.
    year : int
        year of observation.
    month : int
        month of observation.
    day : int
        day of observation.
    targets : pandas df
        List of targets to process.
    plot_it : bool, optional
        To plot the results. The default is False.

    Returns
    -------
    targets_info : pandas df
        Targets with obs. info.

    """

    # date
    year, month, day = get_date(mjd)

    # grab the targets
    stars_alt.target_location(targets=targets)

    # get stars alt
    stars_alt(year=year, month=month, day=day)

    # grab star infos
    alt_min = 25.
    alt_max = 86.5
    airmass_max = 2.5

    targets_info = stars_alt.target_info(star_alt_min=alt_min*u.deg,
                                         star_alt_max=alt_max*u.deg,
                                         star_airmass_max=airmass_max)

    """
    print(targets_info.columns)
    print(targets_info[['target', 'mjd', 'mjd_per_min_p1',
          'mjd_per_max_p1', 'obs_duration [h]']])
    """
    # plot result here
    stars_alt.plot(star_alt_min=alt_min*u.deg,
                   star_alt_max=alt_max*u.deg,
                   star_airmass_max=airmass_max,
                   time_obs=Time(mjd, format='mjd'),
                   hour_min=-6,
                   hour_max=8,
                   fig=fig, ax=ax)

    return targets_info


def make_df(target, ra, dec):

    df = pd.DataFrame(target, columns=['target'])
    df['ra'] = ra
    df['dec'] = dec

    return df


def get_date(mjd):

    ttime = Time('{}'.format(mjd), format='mjd')
    tdate = '{}'.format(ttime.datetime64)
    spa = tdate.split('T')[0].split('-')
    year = int(spa[0])
    month = int(spa[1])
    day = int(spa[2])

    return year, month, day


def target_to_pixel(nside, targets):

    healpixID, pixRA, pixDec = getPix(nside, targets['ra'], targets['dec'])

    df = pd.DataFrame(healpixID, columns=['healpixID'])
    df['pixRA'] = pixRA
    df['pixDec'] = pixDec

    return df


def get_targets(pixels, targets):

    tt = targets[['source_id', 'ra', 'dec']]

    # make all possible combinations pixels/targettarget
    combis = pixels.merge(tt, how='cross')

    print('allo', len(combis))

    combis['deltaRA'] = (combis['pixRA']-combis['ra']) * \
        np.cos(np.deg2rad(combis['pixRA']))
    combis['deltaDec'] = combis['pixDec']-combis['dec']
    combis['dist'] = np.sqrt(combis['deltaRA']**2+combis['deltaDec']**2)

    combis = combis.sort_values(by='dist')

    print(combis[['healpixID', 'source_id', 'dist']])


parser = OptionParser(
    description='Script build an AuxTel survey')

parser.add_option('--dbDir', type=str,
                  default='../DB_Files',
                  help='Data dir [%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--nside', type=int,
                  default=128,
                  help='Healpix nside parameter [%default]')
parser.add_option('--deltaRA', type=float,
                  default=10,
                  help='RA width around pointing center [%default]')
parser.add_option('--deltaDec', type=float,
                  default=10,
                  help='Dec width around pointing center [%default]')
parser.add_option('--fp_level', type=str,
                  default='raft',
                  help='FP granularity level (ccd,raft,sensor) [%default]')
parser.add_option('--targetDir', type=str,
                  default='~/Bureau',
                  help='targets data dir [%default]')
parser.add_option('--targetFile', type=str,
                  default='gaia_source_file_ddf_v0.parquet',
                  help='targets data file [%default]')


opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
nside = opts.nside
deltaRA = opts.deltaRA
deltaDec = opts.deltaDec
fp_level = opts.fp_level
targetDir = opts.targetDir
targetFile = opts.targetFile

# band index
bbands = dict(zip('ugrizy', [1, 2, 3, 4, 5, 6]))

# StarAltTime instance
stars_alt = StarAltTime()

# Pixels_in_FP instance
pix_in_fp = Pixels_in_FP(nside, deltaRA, deltaDec, level=fp_level)

# load targets

targets = pd.read_parquet('{}/{}'.format(targetDir, targetFile))

print(targets)

print(targets.columns)

target_pixels = target_to_pixel(nside, targets)

print(target_pixels)

df_var = ['healpixID', 'pixRA', 'pixDec', 'raft']

if fp_level == 'ccd':
    df_var += ['ccd']

if fp_level == 'sensor':
    df_var += ['ccd', 'sensor']

# load the data

fName = '{}/{}.npy'.format(dbDir, dbName)

obs = np.load(fName)

nnights = len(np.unique(obs['night']))

print(nnights)
# loop on nights

for night in range(1, nnights+1):
    idx = obs['night'] == night
    sel_night = obs[idx]

    tt = np.unique(sel_night['target_name']).tolist()
    tt = list(filter(None, tt))

    list_dd = list(filter(lambda x: 'DD' in x, tt))

    # night with no ddf
    if len(list_dd) == 0:
        continue

    print(tt)
    print(list_dd)

    # select only observations with these DDFs
    idxa = np.in1d(sel_night['target_name'], list_dd)

    sel_dd = sel_night[idxa]

    # sort by mjd
    sel_dd = np.sort(sel_dd, order=['mjd'])
    for dd in sel_dd:
        RA = dd['RA']
        Dec = dd['Dec']
        band = dd['filter']
        mjd = np.round(dd['mjd'], 3)
        field = dd['target_name'].split(':')[-1]

        # get pixels in FP
        ppixels = pix_in_fp(dd, RA, Dec)

        # pix_in_fp.plot_pixels_in_FP(ppixels)

        ppixels = ppixels[df_var]

        # grab nearest targets

        targets_nearest = get_targets(ppixels, targets)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))

        ppixels['color'] = bbands[band]
        comment = '{},night={},mjd={},{}-band'.format(field, night, mjd, band)
        target_pixels['color'] = 7
        all_pixels = pd.concat((ppixels, target_pixels))
        plotMollview(all_pixels, ax[0][0], fig, night, nside, comment, n=7)

        # grab the targets
        targets = make_df([field], [RA], [Dec])

        # plot the targets
        rr = process_night(stars_alt, mjd, targets, fig, ax[0][1])

        # plt.tight_layout
        plt.show()

    print(test)
