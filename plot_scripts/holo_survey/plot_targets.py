#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:40:52 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


def plot_data(tt_dist):
    """
    Function to plot data infos

    Parameters
    ----------
    tt_dist : pandas df
        Data to plot.

    Returns
    -------
    None.

    """

    fields = tt_dist['field'].unique()

    fig, ax = plt.subplots(figsize=(12, 8))
    for field in fields:
        idx = tt_dist['field'] == field
        sel = tt_dist[idx]
        ax.hist(sel['dist_field[deg]'], histtype='step')

    ax.grid(visible=True)
    ax.set_xlabel(r'dist [deg]')
    ax.set_ylabel('Number of entries')

    tt_nf = tt_dist.groupby(['field']).apply(
        lambda x: pd.DataFrame({'ntargets': [len(x)]}), include_groups=False).reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    tt_nf = tt_nf.sort_values(by=['ntargets'])
    ax.plot(tt_nf['ntargets'], tt_nf['field'],
            color='k', marker='o', linestyle='None')
    ax.set_xlabel(r'# targets')
    ax.grid(visible=True)

    fig, ax = plt.subplots(figsize=(15, 9))
    tt_dist = tt_dist.sort_values(by=['sp_type_orig'])
    ax.plot(tt_dist['sp_type_orig'], tt_dist['field'],
            color='k', marker='o', linestyle='None')
    ax.grid(visible=True)
    ax.tick_params(axis='x', labelrotation=20., labelsize=10)
    print(tt_dist['sp_type_orig'].unique())

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(tt_dist['field'], tt_dist['g_mag'],
            color='k', marker='o', linestyle='None')
    """
    ax.plot(tt_dist['field'], tt_dist['I'],
            color='r', marker='o', linestyle='None')
    """
    ax.grid(visible=True)
    ax.set_ylabel(r'g [mag]')
    # ax.tick_params(axis='x', labelrotation=20., labelsize=15)

    fig, ax = plt.subplots(figsize=(12, 8))
    for field in fields:
        idx = tt_dist['field'] == field
        sel = tt_dist[idx]
        ax.hist(sel['dist_star[arcmin]'], histtype='step')

    ax.grid(visible=True)
    ax.set_xlabel(r'dist nearest star [\''']')
    ax.set_ylabel('Number of entries')

    plt.show()


def plot_target(sky_map, target, theDir):
    """
    Function to plot target star map

    Parameters
    ----------
    sky_map : pandas df
        Sky map around the target.
    target : int
        target id.
    theDir : str
        Dir with png file for the target.

    Returns
    -------
    None.

    """

    idx = sky_map['main_target'] == ' '.join(target.split('_'))

    print(sky_map['main_target'])
    sel_sky = sky_map[idx]

    print(sel_sky.columns)
    simbad = Simbad()
    simbad.add_votable_fields('sp_type', 'sp_qual', 'G')
    result_table = simbad.query_object(target)

    print(result_table)
    ra_ref = result_table['ra'].value[0]
    dec_ref = result_table['dec'].value[0]

    imgName = 'map_{}.png'.format(target)
    img = np.asarray(Image.open('{}/{}'.format(theDir, imgName)))

    # print(repr(img))

    # imgplot = plt.imshow(img)

    imagebox = OffsetImage(img, zoom=0.35)

    ab = AnnotationBbox(imagebox, (0.4, 0.6))
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    """
    sel_sky['ra_target'] = ra_ref
    sel_sky['dec_target'] = dec_ref
    sel_sky['dra_new'] = (sel_sky['ra']-ra_ref)*np.cos(np.deg2rad(dec_ref))
    sel_sky['ddec_new'] = (sel_sky['dec']-dec_ref)
    sel_sky['dist_new[arcmin]'] = 60. * \
        np.sqrt(sel_sky['dra_new']**2+sel_sky['ddec_new']**2)

    
    print(sel_sky[['ra_target', 'dec_target', 'ra',
          'dec', 'otype','sp_type','dra', 'ddec', 'dist_star[arcmin]', 'dist_new[arcmin]', 'G']])
    """

    print(sel_sky[['ra', 'dec', 'otype', 'sp_type', 'G']])
    ax[0, 0].add_artist(ab)
    ax[0, 0].set_axis_off()

    ax[1, 0].plot(sel_sky['ra'], sel_sky['dec'], 'b*')

    ax[1, 0].plot(result_table['ra'], result_table['dec'], 'rP')

    ax[1, 0].grid(visible=True)
    ax[1, 0].set_xlabel(r'RA [deg]')
    ax[1, 0].set_ylabel(r'Dec [deg]')

    radius = 1./60

    circle1 = plt.Circle((ra_ref, dec_ref), radius, color='r', fill=False)
    ax[1, 0].add_patch(circle1)
    tt = SkyCoord(sel_sky['ra'],
                  sel_sky['dec'], frame='icrs', unit='deg')

    # tick_labels = ax[1, 0].xaxis.major.formatter.seq
    # ax[1, 0].xaxis.major.formatter.seq = tt.ra.to_string(u.hour)
    # ax[1, 0].xaxis.set_major_locator(tt.ra.to_string(u.hour))
    # ax[1, 0].set_xticklabels(tt.ra.to_string(u.hour))
    # ax[1, 0].tick_params(axis='x', labelrotation=20.)
    # rr = tt.to_string('hmsdms')

    """
    print(tt[0])
    ra_h = tt.ra.to_string(u.hour)
    ra_h = tt.ra.hour
    ax[1, 0].plot(ra_h, sel_sky['dec'], 'ko')
    """
    idxb = sel_sky['G'] < 900.
    sel_sky_mag = sel_sky[idxb]
    """
    ax[1, 1].hist(sel_sky[idxb]['G'], histtype='step')
    ax[1, 1].hist(result_table['G'], histtype='step')
    """
    ax[0, 1].plot(sel_sky_mag['dist[arcmin]'], sel_sky_mag['G'], 'ko')
    ax[0, 1].grid(visible=True)
    ax[0, 1].set_xlabel(r'distance [arcmin]')
    ax[0, 1].set_ylabel(r'G [mag]')

    p = ax[1, 1].scatter(sel_sky_mag['ra'], sel_sky_mag['dec'],
                         c=sel_sky_mag['G'], cmap='viridis')
    fig.colorbar(p, ax=ax[1, 1], orientation='vertical', label='G [mag]')
    ax[1, 1].plot(result_table['ra'], result_table['dec'], 'rP')
    circle2 = plt.Circle((ra_ref, dec_ref), radius, color='r', fill=False)
    ax[1, 1].add_patch(circle2)
    ax[1, 1].grid(visible=True)
    ax[1, 1].set_xlabel(r'RA [deg]')
    ax[1, 1].set_ylabel(r'Dec [deg]')

    plt.tight_layout()
    plt.show()


def ana_sky_map(grp, skymap, radius=10):
    """
    Function to analyze the sky map around  target

    Parameters
    ----------
    skymap : pandas df
        sky map.
    target : str
        target.
    radius : float, optional
        radius arouns the target (in arcsec). The default is 10.

    Returns
    -------
    res : pandas df
        nstars inside radius+mag.

    """

    target = grp.name[1]
    idx = sky_map['main_target'] == ' '.join(target.split('_'))

    # print(sky_map['main_target'])
    sel_sky = pd.DataFrame(sky_map[idx])

    # print(sel_sky.columns)

    sel_sky['dist[arcsec]'] = sel_sky['dist[arcmin]']*60.  # distance in arcsec

    # print(sel_sky)

    idxb = sel_sky['dist[arcsec]'] <= radius

    sel_stars = sel_sky[idxb]

    mag_G = -999.
    fmax_type = 'NA'
    if len(sel_stars) >= 1:
        print(sel_stars['r'])
        mag_G = sel_stars['flux_max'].max()
        print(sel_stars[['flux_max', 'flux_max_type']])

    ddict = {}
    ddict['main_target'] = [target]
    ddict['n_stars'] = [len(sel_stars)]
    ddict['flux_max'] = [mag_G]

    res = pd.DataFrame.from_dict(ddict)

    return res


def load_gaia_stars(fis, params, j=0, output_q=None):
    """
    Function to load Gaia stars using multiproc

    Parameters
    ----------
    fis : list(str)
        List of files to load.
    params : dict
        Parameters.
    j : int, optional
        internal tag for multiproc. The default is 0.
    output_q : multiprocessing queue, optional
        where to load the results. The default is None.

    Returns
    -------
    pandas df
        Output data.

    """

    df = pd.DataFrame()

    for fi in fis:
        tt = pd.read_hdf(fi)
        idx = tt['parallax']/tt['parallax_error'] > 10.
        df = pd.concat((df, tt[idx]))
        del tt
    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df


def load_gaia_stars_multiproc(theDir='../gaia_files',
                              gaiadr='gaiadr3',
                              catDir='gold_sample_oba_gaia_source'):
    """
    Function to load Gaia stars

    Parameters
    ----------
    theDir : str, optional
        Main data dir. The default is '../gaia_files'.
    gaiadr : str, optional
        gaia dir. The default is 'gaiadr3'.
    catDir : str, optional
        cat dir. The default is 'gold_sample_oba_gaia_source'.

    Returns
    -------
    df : pandas df
        Loaded data.

    """

    fName = '{}/{}/{}/*.hdf5'.format(theDir, gaiadr, catDir)

    fis = list(glob.glob(fName))

    params = {}

    from sn_tools.sn_utils import multiproc
    df = multiproc(fis, params, load_gaia_stars, nproc=8)

    return df


parser = OptionParser(
    description='Script to analyse (Gaia) stars matching DDFs')

parser.add_option("--theDirs", type=str,
                  default='../sky_map_holo_A_stars,../sky_map_holo_F_stars',
                  help="file directory [%default]")
parser.add_option("--dbName", type=str,
                  default='baseline_v4.3.1_10yrs',
                  help="dbName directory [%default]")

opts, args = parser.parse_args()

theDirs = opts.theDirs.split(',')
dbName = opts.dbName


df = pd.DataFrame()
for theDir in theDirs:
    fName = '{}/{}/targets.hdf5'.format(theDir, dbName)
    dfa = pd.read_hdf(fName)
    df = pd.concat((df, dfa))

# select sptypes
list_sptypes = ['F9VFe-0.8CH-0.5', 'kA2hA6mF2']
idx = df['sp_type_orig'].isin(list_sptypes)

df = pd.DataFrame(df[~idx])

print(df.columns)
plot_data(df)

print(test)

sky_map = pd.read_hdf('{}/sky_map_summary.hdf5'.format(theDir))

pngs = glob.glob('{}/map*.png'.format(theDir))

targets = pd.read_hdf('{}/targets.hdf5'.format(theDir))

df = targets.groupby(['field', 'target']).apply(
    lambda x: ana_sky_map(x, sky_map), include_groups=False).reset_index()

print(df.columns)

plot_data(df)
print(test)

gaiaDir = '~/Bureau'
gaiaFile = 'gaia_source_file_ddf_v0.parquet'
gaiaDir = 'notebooks'
gaiaFile = 'A_stars.parquet'
gaia_stars = pd.read_parquet('{}/{}'.format(gaiaDir, gaiaFile))


print(gaia_stars.columns)
gaia_stars_cat = load_gaia_stars_multiproc()
print(gaia_stars_cat.columns)
fig, ax = plt.subplots()
gaia_stars['BP'] = gaia_stars['phot_bp_mean_mag'] - \
    gaia_stars['phot_rp_mean_mag']
ax.plot(gaia_stars['BP'], gaia_stars['phot_g_mean_mag'], 'r*')
gaia_stars_cat['BP'] = gaia_stars_cat['phot_bp_mean_mag'] - \
    gaia_stars_cat['phot_rp_mean_mag']
ax.plot(gaia_stars_cat['MG'], gaia_stars_cat['BP'],  'ko')
print(len(gaia_stars_cat))
plt.show()

"""
df = pd.DataFrame()
for png in pngs:
    starid = png.split('map_')[-1].split('.png')[0]
    # plot_target(sky_map, starid, theDir)
    dfb = ana_sky_map(sky_map, starid)
    df = pd.concat((df, dfb))

print(df)
"""
