#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:42:18 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from sn_plotter_metrics.plot4metric import plot_pixels


def plot_cadence(datam, figtitle='', varx='season',
                 legx='season', vary='cadence',
                 legy='cadence [day]', plot_mean=True):

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(figtitle)

    idx = datam[vary] > 0
    data = datam[idx]

    ax.plot(data[varx], data[vary], 'k.', ms=8, mfc='None')

    if plot_mean:
        vv = data.groupby([varx])[vary].mean().reset_index()
        vvb = data.groupby([varx])[vary].std()
        vstd = f'{varx}_std'
        vv[vstd] = vvb.to_list()
        ax.errorbar(vv[varx], vv[vary],
                    yerr=vv[vstd], color='r')

    ax.set_xlabel(r'{}'.format(legx))
    ax.set_ylabel(r'{}'.format(legy))
    ax.set_ylim([0, None])
    ax.grid(visible=True)


def multiplot_season(sel, varx, legx, vary, legy):

    fields = sel['field'].unique()

    bands = 'ugrizy'
    for field in fields:
        idx = sel['field'] == field
        selb = sel[idx]
        plot_cadence(selb, figtitle=field, varx=varx,
                     legx=legx, vary=vary, legy=legy)
        for b in bands:
            vvary = f'{vary}_{b}'
            figtitle = f'{field} - {b} band'
            plot_cadence(selb, figtitle=figtitle, varx=varx,
                         legx=legx, vary=vvary, legy=legy)


def multiplot_dist(sel, yvar='cadence', yleg='cadence [day]'):

    fields = sel['field'].unique()

    bands = 'ugrizy'
    seasons = range(1, 11, 1)
    colors = ['r', 'b', 'k', 'g', 'orange']*2
    lstyles = ['solid']*5+['dashed']*5
    mstyles = ['o', 's', 'P', 'D', 'x']*5
    cols = dict(zip(seasons, colors))
    lstys = dict(zip(seasons, lstyles))
    marks = dict(zip(seasons, mstyles))
    for field in fields:
        idx = sel['field'] == field
        selb = sel[idx]
        seasons = selb['season'].unique()
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.subplots_adjust(right=0.80)
        fig.suptitle(field)
        for seas in range(1, 11):
            idxb = selb['season'] == seas
            selc = selb[idxb]
            print('plotting season', seas)
            plot_pixels(selc, yvar=yvar,
                        yleg=yleg, fig=fig, ax=ax, showIt=False,
                        color=cols[seas], ls=lstys[seas], marker=marks[seas],
                        label=f'season {seas}', ms=12, markevery=10,
                        rebin=True, smoothIt=True, distval='dist')

        ax.grid(visible='True')
        ax.legend(bbox_to_anchor=(0.99, 0.8),
                  ncol=1, frameon=False, fontsize=15)
        # ax.set_xlim([0, None])


parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbName', type=str, default='test_newb',
                  help='dbName to process [%default]')
parser.add_option('--dbDir', type=str, default='../test_metric',
                  help='dbDir of the OS to process [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName

fName = '{}/{}.hdf5'.format(dbDir, dbName)

df = pd.read_hdf(fName)

print(df.columns)

idx = df['season'] > 0
idx &= df['season'] < 11
idx &= df['cadence'] > 0.
sel = df[idx]

varx = 'season'
legx = 'season'
vary = 'cadence'
legy = 'cadence [day]'

# multiplot_season(sel, varx, legx, vary, legy)
multiplot_dist(sel)
multiplot_dist(sel, yvar='nvisits', yleg=r'N$_{visits}$')

plt.show()
