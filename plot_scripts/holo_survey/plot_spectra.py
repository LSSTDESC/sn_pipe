#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:29:53 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import glob
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


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


def load_spectra(spectraDir, source_id):
    """
    Function to load spectra from source_id

    Parameters
    ----------
    spectraDir : str
        spectra directory.
    source_id : int
        source id.

    Returns
    -------
    df : pandas df
        resulting data.

    """

    sName = 'XP_SAMPLED-Gaia DR3 {}.csv'.format(source_id)
    fName = '{}/{}'.format(spectraDir, sName)

    df = pd.read_csv(fName)

    return df


def plot_spectra(spectraDir, source_id, sp_type, fig=None, ax=None):
    """
    Function to plot spectra

    Parameters
    ----------
    spectraDir : str
        spectra directory.
    source_id : int
        source id.
    sp_type : str
        spectral type.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    df = load_spectra(spectraDir, source_id)

    ax.errorbar(df['wavelength'], df['flux'],
                yerr=df['flux_error'], label='{}'.format(sp_type))


parser = OptionParser(
    description='Script to plot spectra from  (Gaia) stars matching DDFs')

parser.add_option("--starDir", type=str,
                  default='../holo_survey',
                  help="file directory for Gaia stars [%default]")
parser.add_option("--dbName", type=str,
                  default='baseline_v4.3.1_10yrs',
                  help="dbName directory [%default]")
parser.add_option("--fName", type=str,
                  default='targets.hdf5',
                  help="targets file [%default]")
parser.add_option("--spectraDir", type=str,
                  default='../gaia_spectra/A_stars',
                  help="file directory for Gaia spectra [%default]")

opts, args = parser.parse_args()

starDir = opts.starDir
dbName = opts.dbName
fName = opts.fName
spectraDir = opts.spectraDir

# load stars
fullName = '{}/{}/{}'.format(starDir, dbName, fName)
df = pd.read_hdf(fullName)

# grab spectra list
ll_spectra = get_spectra_ids(spectraDir)

print(df['source_id'])

idx = df['source_id'].isin(ll_spectra)

sel = df[idx]

print(sel)
print('nspectra', len(df['source_id'].unique()),
      len(sel['source_id'].unique()))

print(sel[['source_id', 'sp_type_orig']])

fig, ax = plt.subplots(figsize=(12, 8))


sources_id = []
while 1:
    answer = input('Spectra to plot (source_id)? ')

    if answer == 'exit':
        break

    if answer == '':
        continue

    source_id = int(answer)
    sources_id.append(source_id)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 8))
    # grab the spectral type
    for vv in sources_id:
        idx = sel['source_id'] == vv
        sp_type = sel[idx]['sp_type_orig'].to_list()[0]
        plot_spectra(spectraDir, vv, sp_type, fig=fig, ax=ax)

    ax.set_xlabel('wavelength [nm]')
    ax.set_ylabel('flux [W nm$^{-1}$ m$^{-2}$]')
    ax.legend()
    ax.grid(visible=True)
    plt.show(block=False)
