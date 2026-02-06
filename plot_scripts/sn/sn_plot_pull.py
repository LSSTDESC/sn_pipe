#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:46:58 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_analysis.sn_tools import load_data, complete_df, pull_it
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_pull_hist(data, fig=None, ax=None, figtit=''):

    if fig is None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    if figtit != '':
        fig.suptitle(figtit)

    ipos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    bins = np.arange(-5, 5, 0.5)

    for i, vv in enumerate(['x1', 'color', 'mb', 'daymax']):
        pp = ipos[i]
        ax[pp[0], pp[1]].hist(data['pull_{}'.format(vv)],
                              bins, histtype='step')

    for i in range(2):
        for j in range(2):
            ax[i, j].grid(visible=True)

    plt.show()


def plot_pull_vs(data, fig=None, ax=None, figtit='',varx='sigma_x1'):

    if fig is None:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    if figtit != '':
        fig.suptitle(figtit)

    ipos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    bins = np.arange(-5, 5, 0.5)

    for i, vv in enumerate(['x1', 'color', 'mb', 'daymax']):
        pp = ipos[i]
        ax[pp[0], pp[1]].plot(
            data[varx], data['pull_{}'.format(vv)], 'ko')
        
        ax[pp[0], pp[1]].set_xlabel(r'{}'.format(varx))
        ax[pp[0], pp[1]].set_ylabel(r'pull {}'.format(vv))
        

    for i in range(2):
        for j in range(2):
            ax[i, j].grid(visible=True)

    plt.show()


parser = OptionParser(description='Script to plot SN parameter pulls')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_newatm_lc_coadd_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v5.0.0_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
timescale = opts.timescale
fields = opts.fields.split(',')

# load data

df = load_data(dbDir, dbName, runType)


# complete data
df = complete_df(df)
df['SNR_red'] = df['SNR_i']+df['SNR_z']+df['SNR_y']

print('before', len(df))
idx = df['fitstatus'] == 'fitok'

df = df[idx]
# rint(df)

print('after', len(df))
# estimate pull
df = pull_it(df)

print(df.columns.to_list())

idx = df['field'] == 'COSMOS'
idx &= df['healpixID'] == 108958

sel = df[idx]
seasons = sel['season'].unique()
seasons = [3]

ccols = ['healpixID', 'z', 'x1', 'color', 'daymax', 'x0', 'season', 'epsilon_x0',
         'epsilon_x1', 'epsilon_color', 'epsilon_daymax', 'SNID',
         'minRFphase', 'minRFphaseQual', 'maxRFphase', 'maxRFphaseQual']
for seas in seasons:
    figtit = 'season {}'.format(seas)
    idxb = sel['season'] == seas
    selb = sel[idxb]
    plot_pull_hist(selb, figtit=figtit)
    plot_pull_vs(selb, figtit=figtit)
    
    ido = selb['pull_color'] < -4.
    seld = selb[ido]
    if len(seld) >= 1:
        snids = seld['SNID'].to_list()
        # select df data
        idf = df['SNID'].isin(snids)
        seldf = df[idf]
        seldf[ccols].to_hdf('simuparams_COSMOS.hdf5', key='simuparams')
