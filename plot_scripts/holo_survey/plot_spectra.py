#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:29:53 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import glob
from optparse import OptionParser


def get_spectra_ids(spectraDir):
    """
    Function to grab the list of spectra (source_id)

    Parameters
    ----------
    spectraDir : TYPE
        DESCRIPTION.

    Returns
    -------
    ll_gaia_ids : TYPE
        DESCRIPTION.

    """

    tt = glob.glob('{}/*.csv'.format(spectraDir))

    ll_gaia_ids = []
    for vv in tt:
        spl = vv.split('.csv')[0].split('DR3')[1]
        ll_gaia_ids.append(spl)

    ll_gaia_ids = list(map(int, ll_gaia_ids))

    return ll_gaia_ids


parser = OptionParser(
    description='Script to plot spectra from  (Gaia) stars matching DDFs')

parser.add_option("--starDir", type=str,
                  default='../holo_survey',
                  help="file directory for Gaia stars [%default]")
parser.add_option("--dbName", type=str,
                  default='baseline_v4.3.1_10yrs',
                  help="dbName directory [%default]")
parser.add_option("--spectraDir", type=str,
                  default='../gaia_spectra/A_stars',
                  help="file directory for Gaia spectra [%default]")

opts, args = parser.parse_args()

theDirs = opts.theDirs.split(',')
dbName = opts.dbName
spectraDir = opts.spectraDir

# grab spectra list
ll_spectra = get_spectra_ids(spectraDir)


print(df['source_id'])

idx = df['source_id'].isin(ll_spectra)

sel = df[idx]

print(sel)
print('nspectra', len(df['source_id'].unique()),
      len(sel['source_id'].unique()))
