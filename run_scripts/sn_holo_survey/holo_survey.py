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
from astropy.visualization import astropy_mpl_style, quantity_support

plt.style.use(astropy_mpl_style)
quantity_support()

plt.rcParams["axes.labelsize"] = "medium"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.minor.size"] = 5
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"

# plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['axes.facecolor'] = 'blue'
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
# the line width around the marker symbol
plt.rcParams['lines.markeredgewidth'] = 0.3
plt.rcParams['lines.markersize'] = 5  # markersize, in points
plt.rcParams['grid.alpha'] = 0.75  # transparency, between 0.0 and 1.0
plt.rcParams['grid.linestyle'] = '-'  # simple line
plt.rcParams['grid.linewidth'] = 0.4  # in points
plt.rcParams['font.size'] = 13


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

    plt.axes(axa)
    hp.mollview(hpxmap, nest=True, cmap=cmap,
                min=xmin, max=n, norm=norm, cbar=False,
                title=comment, hold=True, badcolor='white', xsize=800)

    hp.graticule(verbose=False)

    axb = axa.inset_axes([39., -7.2, 7, 5], projection='mollweide')

    axa.indicate_inset_zoom(axb, edgecolor="black")


def process_night(stars_alt, mjd, targets, fig, ax, plt=None):
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

    # plot result here
    stars_alt.plot(star_alt_min=alt_min*u.deg,
                   star_alt_max=alt_max*u.deg,
                   star_airmass_max=airmass_max,
                   time_obs=Time(mjd, format='mjd'),
                   hour_min=-6,
                   hour_max=8,
                   fig=fig, ax=ax, myplt=plt)

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

    print('hello', len(targets), targets.columns)

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

    tt = combis.groupby(['source_id']).apply(
        lambda x: min_dist_source(x), include_groups=False).reset_index()

    print(tt)

    print(len(tt))

    idx = tt['dist'] <= 5

    res = tt[idx]

    res['target'] = res['source_id']

    vv = ['source_id', 'healpixID', 'pixRA', 'pixDec',
          'raft', 'deltaRA', 'deltaDec', 'dist', 'target']

    res = res[vv]

    ll = res['source_id'].to_list()

    idx = targets['source_id'].isin(ll)

    sel_targets = targets[idx]

    sel_targets = sel_targets.merge(res, left_on=['source_id'],
                                    right_on=['source_id'], suffixes=['', ''])
    return sel_targets


def min_dist_source(grp):

    grp = grp.sort_values(by=['dist'])

    print('lll', grp[:1])
    return pd.DataFrame(grp[:1])


def plot_flat(pixels_FP, target_pixels, ra, dec,
              width_ra=3.,
              width_dec=3., fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(pixels_FP['pixRA'], pixels_FP['pixDec'], marker='o', color='r',
            mfc='None', linestyle='None', label='VRO FP')
    ax.plot(target_pixels['pixRA'],
            target_pixels['pixDec'], 'b*', label='target')

    ax.set_xlim([ra-width_ra, ra+width_ra])

    ax.set_ylim([dec-width_dec, dec+width_dec])

    ax.grid(visible=True)

    ax.set_xlabel(r'Right Ascension [deg]')
    ax.set_ylabel(r'Declination [deg]')

    ax.legend(loc='upper right', fontsize=10, frameon=False)
    """,
              bbox_to_anchor=(1.3, 0.8), fontsize=10,
              frameon=False)
    """


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


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

targets = targets.rename(
    columns={'phot_g_mean_mag': 'g_mag', 'phot_variable_flag': 'var_flag'})

targets = targets.round({'ra': 2, 'dec': 2, 'g_mag': 2})
targets['var_flag'] = targets['var_flag'].str.replace(
    'NOT_AVAILABLE', 'NA')


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

        print(targets_nearest.columns)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(
            10, 10))

        fig.subplots_adjust(wspace=0.4)

        ppixels['color'] = bbands[band]
        comment = '{},night={},mjd={},{}-band'.format(field, night, mjd, band)
        target_pixels['color'] = 7
        all_pixels = pd.concat((ppixels, target_pixels))
        plotMollview(all_pixels, ax[0][0], fig, night, nside, comment, n=7)

        # plot pixels (flat mode)
        plot_flat(ppixels, target_pixels, RA, Dec, fig=fig, ax=ax[1][0])

        # grab the targets
        targets_b = make_df([field], [RA], [Dec])

        # plot the targets
        rr = process_night(stars_alt, mjd, targets_nearest, fig, ax[0][1], plt)
        # set_size(5, 5, ax=ax[0][1])

        # print target results

        targets_nearest = targets_nearest.sort_values(by=['dist'])

        tp = ['source_id', 'ra', 'dec', 'g_mag', 'var_flag']

        axc = ax[1][1]

        axc.axis('off')

        rr = pd.DataFrame(targets_nearest[tp])
        rr['g_mag'] = rr['g_mag'].astype(str)
        print(rr)
        # rr.style.hide(axis='index')
        # rr = rr.reset_index()
        # rr = rr.to_string(index=False)
        # rr = rr.set_index('source_id')
        table = pd.plotting.table(axc, rr,
                                  loc='center', cellLoc='center',
                                  colWidths=[0.35]+[0.1]*4)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.8, 1.8)
        """
        axc.table(cellText=rr.values, colLabels=rr.keys(),
                  loc='center', cellLoc='center',
                  colWidths=[0.35]+[0.1]*4, fontsize=20, scale=(2, 2))
        """
        plt.tight_layout
        plt.show()

    print(test)
