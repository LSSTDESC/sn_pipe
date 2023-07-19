#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:21:12 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from optparse import OptionParser
import numpy as np
from sn_plotter_cosmology import plt
from matplotlib.patches import Ellipse

parser = OptionParser(
    description='Script to plot confidence ellipse')

parser.add_option('--dbDir', type=str,
                  default='../cosmo_fit_WFD_sigmaC_1.0',
                  help='Dir ofor cosmo results [%default]')
parser.add_option('--dbName', type=str, default='DDF_DESC_0.80_SN',
                  help='OS to process [%default]')
parser.add_option('--budget_DD', type=str, default='0.07',
                  help='DD budget [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
budget = opts.budget_DD

# load data
fullName = '{}/cosmo_{}_{}.hdf5'.format(dbDir, dbName, budget)

df = pd.read_hdf(fullName)

# select only config with MoM > 0.
idx = df['MoM'] > 0.
df = df[idx]

# get eigenvalues (a,b) as ellipse axis
delta_chi2 = 6.17  # 95.4% C.L.
df['A'] = 0.5*(df['Cov_w0_w0_fit']+df['Cov_wa_wa_fit'])
df['B'] = 0.25*(df['Cov_w0_w0_fit']-df['Cov_wa_wa_fit'])**2 + \
    df['Cov_wa_w0_fit']**2
df['a'] = df['A']+np.sqrt(df['B'])
df['b'] = df['A']-np.sqrt(df['B'])
df['a'] *= np.sqrt(delta_chi2)
df['b'] *= np.sqrt(delta_chi2)
df['angle'] = 0.5*np.arctan((2.*df['Cov_wa_w0_fit']) /
                            (df['Cov_w0_w0_fit']-df['Cov_wa_wa_fit']))
print(df[['a', 'b', 'angle']])

seasons = df['season'].unique()
for season in seasons:
    idc = df['season'] == season
    sel = df[idc]

    fig, ax = plt.subplots()
    ax = plt.gca()
    max_x = []
    max_y = []
    min_x = []
    min_y = []
    for i, row in sel.iterrows():
        x = row['w0_fit']
        y = row['wa_fit']
        a = row['a']
        b = row['b']
        angle = np.rad2deg(row['angle'])
        print('all', x, y, a, b, angle, season)
        ell = Ellipse(xy=(x, y),
                      width=b, height=a,
                      angle=angle, color='k')
        ell.set_facecolor('None')
        ax.add_artist(ell)
        ax.scatter(x, y, c='red', s=3)

        delta_x = np.abs(a*np.cos(row['angle']))
        delta_y = np.abs(b*np.cos(row['angle']))
        max_x.append(x+delta_x)
        max_y.append(y+delta_y)
        min_x.append(x-delta_x)
        min_y.append(y-delta_y)

        print('ttt', y, delta_y, row['angle'])
        if i == 0:
            break

    print('aooo', np.max(max_x), np.max(max_y), np.min(min_x), np.min(min_y))
    ax.set_xlim(np.min(min_x), np.max(max_x))
    ax.set_ylim(np.min(min_y), np.max(max_y))
    # ax.set_xlim([-maxx, np.max(max_x)])
    # ax.set_ylim([None, np.max(max_y)])
    fig.tight_layout()
plt.show()
