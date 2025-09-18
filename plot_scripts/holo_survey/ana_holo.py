#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 08:45:43 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pprint
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
import astropy.units as u
from urllib.parse import urlencode
from astropy.io import fits
from astropy.table import unique, Table
import aplpy
import numpy as np
import time
from sn_tools.sn_utils import clean_level
from sn_tools.sn_io import checkDir

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


def get_ra_dec_from_name(nameList=["hd111980", "hd101452",
                                   "hd115169", "hd142331"]):
    """
    function to grab the (ra,dec) of stars corresponding to a list of star ids.

    Parameters
    ----------
    nameList : list(str), optional
        List of stars to process. The default is ["hd111980", "hd101452",
                                     "hd115169", "hd142331"].

    Returns
    -------
    res : pandas df
        output data with the collowing columns: target,ra,dec.

    """

    from astroquery.simbad import Simbad

    r = []
    for starName in nameList:
        result_table = Simbad.query_object(starName)
        """
        print(result_table[0]['ra'])
        ra = ':'.join(result_table[0]['ra'].split(' '))
        dec = ':'.join(result_table[0]['dec'].split(' '))
        """
        ra = result_table[0]['ra']
        dec = result_table[0]['dec']
        coords = '{} {}'.format(ra, dec)
        c = SkyCoord(coords, frame='fk5', unit=(u.deg, u.deg))

        r.append((starName, c.ra.degree, c.dec.degree,
                  c.ra.to_string(u.hour), c.dec.to_string(u.hour)))

    res = pd.DataFrame(r, columns=['target', 'ra', 'dec', 'ra_h', 'dec_h'])
    
    print('iii',res,ra,dec)
    return res


def analyse_map(grp, radius=2):
    """
    Function to analyse the results of the map stars.

    Parameters
    ----------
    grp : pandas df
        Data to process (list of stars).
    radius : int, optional
        Radius for the search. The default is 2 arcmin.

    Returns
    -------
    out_df : pandas df
        Output data.

    """

    # remove duplicates (if any)
    # grp = grp.drop_duplicates()

    # first thing: grab main_target
    # id_main = grp['main_target'].unique()[0]
    id_main = grp.name

    # grab infos for this target
    main_target = get_ra_dec_from_name([id_main])

    # estimate distance wrt star_id
    ra_ref = main_target['ra'].values[0]
    dec_ref = main_target['dec'].values[0]

    print('grp', grp)
    print('ref',ra_ref,dec_ref)
    # search this target in the star list
    idx = np.abs(grp['ra']-ra_ref)*60. < 0.01
    idx &= np.abs(grp['dec']-dec_ref)*60 < 0.01

    print(grp[['main_id']])

    # idx = grp['main_id'] == grp['main_target']

    stars = grp[~idx]
    star_ref = grp[idx]

    """
    out_df = pd.DataFrame([id_main], columns=['target'])

    print(star_ref, star_ref.columns)
    ccols = ['ra', 'dec', 'sp_type',
             'mesdistance.dist', 'mesdistance.unit',
             'G', 'otype', 'ra_deg', 'dec_deg']

    if len(star_ref) == 1:
        out_df[ccols] = star_ref[ccols].values
    else:
        print('eee', grp[ccols].values)
        out_df[ccols] = grp[ccols].values
    """
    # remove planets and starid
    idx = stars['otype'] != 'Planet'
    # idx &= stars['target'] != star_id
    stars = pd.DataFrame(stars[idx])
    # number of stars
    nstars = len(stars)

    """
    out_df['radius_arcmin'] = int(radius)
    out_df['Nobj_radius'] = nstars
    """

    if len(stars) > 0:
        stars['dra'] = (stars['ra']-ra_ref)*np.cos(np.deg2rad(dec_ref))
        stars['ddec'] = (stars['dec']-dec_ref)

        stars['dist_star[arcmin]'] = 60. * \
            np.sqrt(stars['dra']**2+stars['ddec']**2)

        stars = stars.fillna(999.)
        return stars
        print('allo ici', stars[['dra', 'ddec', 'dist_star[arcmin]']])
        stars = stars.fillna(999.)
        # print(stars[['OTYPE', 'dist_star[arcsec]']])
        # nearest star
        idxmin = stars['dist_star[arcmin]'].idxmin()
        # print('nearest stars', stars.loc[idxmin])
        nearest_star = stars.loc[idxmin].to_frame().T

        nn = nearest_star[['dist_star[arcmin]',
                           'flux_max', 'flux_max_type']]
        nn = nn.rename(columns={'flux_max': 'mag_min_dist',
                                'flux_max_type': 'mag_min_dist_type'})
        # out_df[nn.columns] = nn[nn.columns].values.tolist()
        out_df = pd.concat([out_df, nn], axis=1)

        idxmin = stars['flux_max'].idxmin()
        star_high_flux = stars.loc[idxmin].to_frame().T
        """
        if len(stars) == 1:
            star_high_flux = stars.loc[idxmin].to_frame().T
        else:
            star_high_flux = stars.loc[idxmin][:1]
        """
        tt = star_high_flux[['flux_max', 'flux_max_type']]
        tt = tt.rename(columns={'flux_max': 'mag_min',
                                'flux_max_type': 'mag_min_type'})

        # out_df = pd.concat((out_df, tt), axis=1)
        # out_df[tt.columns] = tt[tt.columns].values.tolist()
        out_df = pd.concat([out_df, tt], axis=1)
    else:
        # out_df[['dist_star[arcmin]', 'mag_min_dist', 'mag_min_dist_type']] = 999.
        # out_df[['mag_min', 'mag_min_type']] = 999.
        return pd.DataFrame()

    return -1


def sky_map_summary(df, radius=2):
    """
    Function to get star map infos

    Parameters
    ----------
    df : pandas df
        Data to process.
    radius : int, optional
        Radius for the search window around the ref stars.
        The default is 2 arcsec.

    Returns
    -------
    resb : pandas df
        Output data.

    """

    simbad = Simbad()
    """
    simbad.add_votable_fields(
        "sptype", "flux(B)", "flux(V)", "flux(R)", "flux(I)", "flux(G)",
        "flux(K)", "flux(H)", "flux(u)", "flux(g)", "flux(r)", "flux(i)",
        "flux(z)", "distance", "otype")
    """
    simbad.TIMEOUT = 1000

    # get nearby stars
    stars = get_stars(df, simbad, radius=radius)

    print(stars.columns.tolist())

    ccols = ['main_id', 'ra', 'dec',
             # 'coo_err_maj',
             # 'coo_err_min', 'coo_err_angle',
             # 'coo_wavelength',
             # 'coo_bibcode',
             'K', 'z', 'I', 'i', 'r',
             'B', 'G', 'R', 'V', 'u', 'g', 'H', 'sp_type', 'sp_qual']
    print('resultat', stars[ccols])

    ccols = 'main_target'
    # print(res[ccols], type(res))

    stars = clean_level(stars)
    resb = stars.groupby(by=[ccols]).apply(
        lambda x: analyse_map(x, radius=radius), include_groups=False).reset_index()

    return resb


def simbad_query_info(simbad, llist):
    """
    Function to grab object info from Simbad

    Parameters
    ----------
    simbad: Simbad()
        simbad instance
    llist : list(str)
        List of IDs to prospect.

    Returns
    -------
    res : pandas df
        output data.

    """

    ccols = ['ra', 'dec', 'sp_type', 'sp_qual', 'B', 'V', 'R',
             'I', 'G',
             'K', 'H', 'u', 'g',
             'r', 'i', 'z']
    ccols_flux = ['B', 'V', 'R',
                  'I', 'G',
                  'K', 'H', 'u', 'g',
                  'r', 'i', 'z']
    ccols = ['ra', 'dec', 'sp_type',
             'ra_deg',
             'dec_deg', 'otype']+ccols_flux
    # 'Diameter_diameter', 'Diameter_unit', 'Diameter_error']

    simbad.add_votable_fields('sp_type', 'sp_qual', 'B', 'V',
                              'R', 'I', 'G', 'K', 'H', 'u', 'g',
                              'r', 'i', 'z', 'otype')
    tab = simbad.query_objects(llist)

    tab.remove_columns(['coo_err_maj', 'coo_err_min',
                        'coo_err_angle', 'coo_wavelength', 'coo_bibcode'])
    """
    tab = unique(tab)
    """
    tab.convert_bytestring_to_unicode()
    tab = tab.to_pandas()
    restot = tab.groupby(['main_id']).apply(
        lambda x: x[:1], include_groups=False).reset_index()

    print(restot.columns.to_list(), len(restot))
    """
    restot = restot.drop_duplicates()
    print('hhhh', restot, len(restot), len(llist), llist)

    restot['target'] = llist
    """

    coords = SkyCoord(ra=restot['ra'], dec=restot['dec'],
                      unit=(u.hourangle, u.deg), frame='icrs')

    restot['ra_deg'] = coords.ra.deg
    restot['dec_deg'] = coords.dec.deg

    restot[ccols] = restot[ccols].replace('--', np.nan)

    """
    for ll in llist:
        # search_id = 'Gaia DR3 {}'.format(ll)
        tab = simbad.query_object(ll)
        tab.convert_bytestring_to_unicode()
        # print(tab.columns)
        tabb = tab.to_pandas()
        tabb['target'] = ll
        coords = SkyCoord(ra=tab['RA'], dec=tab['DEC'],
                          unit=(u.hourangle, u.deg), frame='icrs')
        tabb['RA_deg'] = coords.ra.deg
        tabb['Dec_deg'] = coords.dec.deg

        tabb[ccols] = tabb[ccols].replace('--', np.nan)
        restot = pd.concat((restot, tabb[ccols]))
    """
    restot['flux_max'] = restot[ccols_flux].min(axis=1)
    restot['flux_max_type'] = restot[ccols_flux].idxmin(
        axis="columns").to_list()

    return restot


def get_stars(df, simbad, radius=2, idCol='target', RACol='ra', DecCol='dec'):
    """
    Function to grab stars corresponding to a set of targets

    Parameters
    ----------
    df : pandas df
        Target infos (target, ra, dec).
    simbad : Simbad
        Simbad instance.
    radius : intptional
        Radius around the reference stars. default is 2 arcmin.
    idCol : str, optional
        id for the star. The default is 'target'.
    RACol : str, optional
        RA col name. The default is 'ra'.
    DecCol : str, optional
        Dec col name. The default is 'dec'.

    Returns
    -------
    res : pandas df
        output data.

    """

    # grab ids
    main_stars_id = df[idCol].to_list()

    df = df[df['target'].isin(main_stars_id)]
    print('alors', main_stars_id)

    df_main = pd.DataFrame(main_stars_id, columns=['main_target'])
    scrstr = 'SCRIPT_NUMBER_ID'
    df_main[scrstr] = df_main.index+1

    # grab the stars
    simbad = Simbad()
    # print(simbad.list_votable_fields().columns['name'].tolist())
    # simbad.add_votable_fields("otype")
    stars = simbad.query_region(SkyCoord(df[RACol].to_list(),
                                         df[DecCol].to_list(),
                                         unit=(u.deg, u.deg), frame='fk5'),
                                radius=[radius * u.arcmin]*len(df))

    print('stars in radius', stars.columns, len(stars), stars)

    stars = stars.to_pandas()
    # grab star infos
    res = simbad_query_info(simbad, stars['main_id'].tolist())

    # print('aooo', res[['MAIN_ID', scrstr]], len(res))
    print('aooo', res.columns, len(res))
    # res = res.drop(columns=[scrstr])
    ccols = ['main_id']
    res = res.merge(stars[ccols], left_on='main_id',
                    right_on='main_id', suffixes=['', ''])
    print(res[['main_id', 'user_specified_id']])
    # add the main target info
    print('allo', df[['target', 'ra', 'dec']])
    dfb = df[['target', 'ra', 'dec']]
    dfb = dfb.rename(columns={'target': 'main_target',
                     'ra': 'ra_target', 'dec': 'dec_target'})

    resb = res.merge(dfb, how='cross')

    vv_ra = (resb['ra']-resb['ra_target']) * \
        np.cos(np.deg2rad(resb['dec_target']))
    vv_dec = resb['dec']-resb['dec_target']

    resb['dist[arcmin]'] = np.sqrt(vv_ra**2+vv_dec**2)*60

    print(resb[['main_id', 'ra', 'ra_target', 'dist[arcmin]']])

    """
    res = res.merge(df_main, left_on=scrstr, right_on=scrstr,
                    suffixes=['', ''])

    """
    idx = resb['dist[arcmin]'] <= radius  # 3 arcmin

    # get_map(resb[idx], idCol='main_target')
    return resb[idx]


def get_map(query_results, outName='map.png',
            idCol='MAIN_ID', RACol='ra', DecCol='dec', fov=2):
    """
    Function to grab a map

    Parameters
    ----------
    query_results : pandas df
        3 columns required: idCol, RACol, DecCol.
    outName : str, optional
        output png name. The default is 'map.png'.
    idCol : str, optional
        id column. The default is 'MAIN_ID'.
    RACol : str, optional
        RA colname. The default is 'ra'.
    DecCol : str, optional
        Dec colname. The default is 'dec'.
    fov : float, optional
        fov for the plot [in arcmin]. The default is 2.
    Returns
    -------
    None.

    """
    print('alllll', idCol, query_results)
    object_main_id = query_results.iloc[0][idCol]
    # decode('ascii')
    object_coords = SkyCoord(ra=query_results[RACol],
                             dec=query_results[DecCol],
                             unit=(u.hourangle, u.deg), frame='icrs')

    hips = 'DSS'
    hips = '2MASS'
    query_params = {'hips': hips,
                    'object': object_main_id,
                    'ra': object_coords[0].ra.value,
                    'dec': object_coords[0].dec.value,
                    'fov': (fov * u.arcmin).to(u.deg).value,
                    'width': 500,
                    'height': 500
                    }
    url = f'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?{
        urlencode(query_params)}'
    hdul = fits.open(url)
    print(hdul.info())
    print(hdul[0].header)

    gc = aplpy.FITSFigure(hdul)
    gc.show_grayscale()
    gc.show_colorscale()
    gc.add_grid()

    gc.show_markers(object_coords[1:].ra, object_coords[1:].dec,
                    edgecolor='red', marker='s', s=25**2)
    gc.show_markers(object_coords[0].ra, object_coords[0].dec,
                    facecolor='black', marker='+', s=10**2)
    gc.save(outName)


def get_map_target(ddict, outName='map.png',
                   idCol='MAIN_ID', RACol='ra', DecCol='dec', fov=2):
    """
    Function to grab a map

    Parameters
    ----------
    query_results : dict
        3 columns required: idCol, RACol, DecCol.
    outName : str, optional
        output png name. The default is 'map.png'.
    idCol : str, optional
        id column. The default is 'MAIN_ID'.
    RACol : str, optional
        RA colname. The default is 'ra'.
    DecCol : str, optional
        Dec colname. The default is 'dec'.
    fov : float, optional
        fov for the plot [in arcmin]. The default is 2.
    Returns
    -------
    None.

    """
    object_main_id = ddict[idCol]
    # decode('ascii')
    object_coords = SkyCoord(ra=[ddict[RACol]],
                             dec=[ddict[DecCol]],
                             unit=(u.hourangle, u.deg), frame='icrs')

    hips = 'DSS'
    hips = '2MASS'
    query_params = {'hips': hips,
                    'object': object_main_id,
                    'ra': object_coords[0].ra.value,
                    'dec': object_coords[0].dec.value,
                    'fov': (fov * u.arcmin).to(u.deg).value,
                    'width': 1000,
                    'height': 1000
                    }
    url = f'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?{
        urlencode(query_params)}'
    hdul = fits.open(url)
    print(hdul.info())
    print(hdul[0].header)

    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.set_axis_off()
    title = '{} \n {}'.format(ddict['field'], ddict['target'])
    gc = aplpy.FITSFigure(hdul)
    gc.show_grayscale()
    gc.show_colorscale()
    gc.add_grid()
    gc.set_title(title)

    gc.show_markers(object_coords[1:].ra, object_coords[1:].dec,
                    edgecolor='red', marker='s', s=25**2)
    print(object_coords[0].ra, object_coords[0].dec)
    gc.show_markers(object_coords[0].ra, object_coords[0].dec,
                    facecolor='black', marker='o', s=50**2)

    # plt.show()
    gc.save(outName)


def process_chunk_starmap(data, params, j=0, output_q=None):
    """
    Function to process star map using a  set of data (multiprocessing)

    Parameters
    ----------
    data : pandas df
        Data to process.
    params : dict
        parameters.
    j : int, optional
        internal tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        Where the data will be stored. The default is None.

    Returns
    -------
    pandas df
        output data.

    """

    from more_itertools import sliced
    CHUNK_SIZE = params['CHUNK_SIZE']

    index_slices = sliced(range(len(data)), CHUNK_SIZE)

    res = pd.DataFrame()
    for index_slice in index_slices:
        time_ref = time.time()
        chunk = data.iloc[index_slice]
        rr = sky_map_summary(chunk)
        res = pd.concat((res, rr))
        print('done', index_slice, time.time()-time_ref)
        # break
    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def process_starmap(data, chunk_size=200):
    """
    Function to process star map

    Parameters
    ----------
    data : pandas df
        Data to process.
    chunk_size : int, optional
        Chunk size used in multiprocessing (simbad database access).
        The default is 500.

    Returns
    -------
    rb : pandas df
        output data.

    """

    data['target'] = 'Gaia DR3 '+data['source_id'].apply(str)

    params = {}
    params['CHUNK_SIZE'] = chunk_size
    rb = multiproc(data, params, process_chunk_starmap, 1)

    return rb


def get_tt_dist(grp):

    print(grp.columns)

    # grab the number of sources

    tt_dist = grp['target'].unique()

    vvals = ['ra', 'dec', 'pmra', 'pmdec', 'parallax', 'g_mag', 'phot_bp_mean_mag',
             'phot_rp_mean_mag', 'l', 'b', 'var_flag', 'ref_epoch',
             'rv_template_teff', 'teff_gspphot', 'source_id']

    tt = grp[vvals]

    tt = tt.drop_duplicates()
    print(len(tt_dist), len(tt.drop_duplicates()))

    return tt


def plot_data(data):
    tt_dist = data.groupby(['field']).apply(
        lambda x: get_tt_dist(x), include_groups=False).reset_index()

    tt_dist = data.groupby(['field', 'source_id', 'sp_type'])[
        'dist'].mean().reset_index()

    print(tt_dist)

    fields = tt_dist['field'].unique()

    fig, ax = plt.subplots(figsize=(12, 8))
    for field in fields:
        idx = tt_dist['field'] == field
        sel = tt_dist[idx]
        ax.hist(sel['dist'], histtype='step')

    ax.grid(visible=True)
    ax.set_xlabel(r'dist [deg]')
    ax.set_ylabel('Number of entries')

    tt_nf = tt_dist.groupby(['field']).apply(
        lambda x: pd.DataFrame({'ntargets': [len(x)]})).reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    tt_nf = tt_nf.sort_values(by=['ntargets'])
    ax.plot(tt_nf['ntargets'], tt_nf['field'],
            color='k', marker='o', linestyle='None')
    ax.set_xlabel(r'# targets')
    ax.grid(visible=True)

    fig, ax = plt.subplots(figsize=(15, 9))
    tt_dist = tt_dist.sort_values(by=['sp_type'])
    ax.plot(tt_dist['sp_type'], tt_dist['field'],
            color='k', marker='o', linestyle='None')
    ax.grid(visible=True)
    ax.tick_params(axis='x', labelrotation=20., labelsize=15)
    plt.show()


parser = OptionParser(
    description='Script to analyze the holo survey')
parser.add_option('--fileDir', type=str, default='../sn_holo_survey',
                  help='OS file dir [%default]')
parser.add_option('--dbName', type=str, default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--surveyName', type=str, default='holo_survey_mean_pointings',
                  help='survey name [%default]')
parser.add_option('--outDir', type=str, default='../sky_map_holo',
                  help='output dir [%default]')

opts, args = parser.parse_args()

fileDir = opts.fileDir
dbName = opts.dbName
surveyName = opts.surveyName
outDir = '{}/{}'.format(opts.outDir, dbName)

checkDir(outDir)

fName = '{}/{}/{}.hdf5'.format(fileDir, dbName, surveyName)

data = pd.read_hdf(fName)

print(data[['source_id', 'ra', 'dec']])

tt_dist = data.groupby(['field', 'source_id', 'sp_type', 'ra', 'dec'])[
    'dist'].mean().reset_index()
tt_dist['target'] = 'Gaia DR3 '+tt_dist['source_id'].apply(str)
print(tt_dist)
# plot_data(data)

idxt = tt_dist['target'] == ' '.join('Gaia_DR3_2493243363030533632'.split('_'))
#tt_dist = tt_dist[10:11]

tt_dist = tt_dist[idxt]
# get images here

for i, row in tt_dist.iterrows():
    fName = 'map_{}.png'.format(row['target'])
    print(fName.split(' '))
    fName = '_'.join(fName.split(' '))
    get_map_target(row.to_dict(),
                   outName='{}/{}'.format(outDir, fName),
                   idCol='target', fov=5)

res = sky_map_summary(tt_dist, radius=5)

print(res.dtypes)
pprint.pprint(res)

res.to_hdf('{}/sky_map_summary.hdf5'.format(outDir), key='skymap')

print(res, res.columns)
print(res[['main_target', 'dist_star[arcmin]', 'G',
      'sp_type', 'otype', 'flux_max', 'flux_max_type']])
