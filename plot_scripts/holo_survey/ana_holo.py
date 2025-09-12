#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 08:45:43 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from urllib.parse import urlencode
from astropy.io import fits


def get_map(query_results, outName='map.png',
            idCol='MAIN_ID', RACol='RA', DecCol='DEC'):
    """
    Function to grab a map

    Parameters
    ----------
    query_results : pandas df
        3 columns required: idCol, RACol, DecCol.

    Returns
    -------
    None.

    """

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
                    'fov': (3.8 * u.arcmin).to(u.deg).value,
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


parser = OptionParser(
    description='Script to analyze the holo survey')
parser.add_option('--fileDir', type=str, default='../sn_holo_survey',
                  help='OS file dir [%default]')
parser.add_option('--dbName', type=str, default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--surveyName', type=str, default='holo_survey_mean_pointings',
                  help='survey name [%default]')

opts, args = parser.parse_args()

fileDir = opts.fileDir
dbName = opts.dbName
surveyName = opts.surveyName

fName = '{}/{}/{}.hdf5'.format(fileDir, dbName, surveyName)

data = pd.read_hdf(fName)

print(data)


tt_dist = data.groupby(['field']).apply(
    lambda x: get_tt_dist(x), include_groups=False).reset_index()


tt_dist = data.groupby(['field', 'source_id','sp_type'])['dist'].mean().reset_index()

print(tt_dist)


fields = tt_dist['field'].unique()

fig, ax = plt.subplots()
for field in fields:
    idx = tt_dist['field'] == field
    sel = tt_dist[idx]
    ax.hist(sel['dist'], histtype='step')

tt_nf = tt_dist.groupby(['field']).apply(
    lambda x: pd.DataFrame({'ntargets': [len(x)]})).reset_index()

fig, ax = plt.subplots()
tt_nf = tt_nf.sort_values(by=['ntargets'])
ax.plot(tt_nf['ntargets'], tt_nf['field'], 'k.')
ax.set_xlabel(r'# targets')
ax.grid(visible=True)

fig, ax = plt.subplots(figsize=(15,9))
tt_dist = tt_dist.sort_values(by=['sp_type'])
ax.plot(tt_dist['sp_type'],tt_dist['field'],'k.')
ax.grid(visible=True)

plt.show()
