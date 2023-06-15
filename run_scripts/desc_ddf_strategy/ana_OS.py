#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:34:46 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from optparse import OptionParser
from brokenaxes import brokenaxes


def coadd_night(grp, colsum=['numExposures'],
                colmean=['season', 'observationStartMJD']):

    dictout = {}
    for vv in colsum:
        dictout[vv] = [grp[vv].sum()]

    for vv in colmean:
        dictout[vv] = [grp[vv].mean()]

    return pd.DataFrame.from_dict(dictout)


def translate(grp):

    grp = grp.sort_values(by=['season'])
    seasons = grp['season'].unique()

    val = 'observationStartMJD'
    Tmax = grp[grp['season'] == 1][val].max()
    dfi = pd.DataFrame()
    for seas in seasons:
        idx = grp['season'] == seas
        sel = pd.DataFrame(grp[idx])
        Tmin = sel[val].min()
        deltaT = Tmin-Tmax
        if seas == 1:
            deltaT = 0.
        sel.loc[:, val] -= deltaT
        Tmax = sel[val].max()
        dfi = pd.concat((dfi, sel))

    return dfi


parser = OptionParser(description='Script to analyze Observing Strategy')

parser.add_option('--dbDir', type=str, default='../DB_Files',
                  help='OS location dir [%default]')
parser.add_option('--dbName', type=str, default='DDF_DESC_0.80_SN.npy',
                  help='OS to analyze [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName

data = np.load('{}/{}'.format(dbDir, dbName))
df = pd.DataFrame.from_records(data)

dfb = df.groupby(['note', 'night']).apply(
    lambda x: coadd_night(x)).reset_index()

fields = dfb['note'].unique()
print(dfb.columns, fields)

ff = ['DD:COSMOS', 'DD:XMM_LSS',
      'DD:ECDFS', 'DD:EDFS_a',
      'DD:EDFS_b', 'DD:ELAISS1']

mk = dict(zip(ff, ['o']*2+['*']*4))
colors = dict(zip(ff, ['k']*2+['r']*4))
fig, ax = plt.subplots(figsize=(14, 9))

val = 'observationStartMJD'
ymax = 1.05*dfb['numExposures'].max()
xmin = dfb[val].min()
# fig, (ax, ax2) = plt.subplots(1, 2, sharey=True,
#                              facecolor='w', figsize=(14, 9))

for io, field in enumerate(fields):
    idx = dfb['note'] == field
    sel = dfb[idx]
    rr = translate(sel)
    rr = rr.sort_values(by=[val])
    xmax = rr[val].max()
    idx = rr['season'] <= 6
    rra = rr[idx]
    ax.plot(rra[val], rra['numExposures'], marker=mk[field], color=colors[field],
            linestyle='None', mfc='None')

    """
    idx = rr['season'] >= 9
    rrb = rr[idx]
    ax2.plot(rrb[val], rrb['numExposures'], marker=mk[field], color=colors[field],
             linestyle='None', mfc='None')
    """

    xmax = rr[val].max()
    # add seasons
    for seas in rra['season'].unique():
        ii = rra['season'] == seas
        selit = rra[ii]
        seas_min = selit[val].min()
        seas_max = selit[val].max()
        ax.plot([seas_min]*2, [0, ymax], color='k', linestyle='dotted')
        ax.plot([seas_max]*2, [0, ymax], color='k', linestyle='dotted')
        diff = seas_max-seas_min
        ax.text(seas_min+0.1*diff, 0.5*ymax,
                'Season', color='k', fontsize=15)
        ax.text(0.5*(seas_min+seas_max), 0.45*ymax,
                '{}'.format(int(seas)), color='k', fontsize=15)


# ax.set_ylim([0, ymax])
# ax.set_xlim([xmin, xmax])
ax.grid()
plt.show()
