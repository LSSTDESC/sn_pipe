#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:42:18 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from sn_plotter_metrics.plot4metric import multiplot_dist
from sn_plotter_metrics.plot4metric import plotMollview_seasons
import numpy as np
import glob


def plot_var_mean(datam, figtitle='', varx='season',
                  legx='season', vary='cadence',
                  legy='cadence [day]', plot_mean=True):
    """
    Function to make a plot and sumperimpose means

    Parameters
    ----------
    datam : pandas df
        Data to process.
    figtitle : str, optional
        figure title. The default is ''.
    varx : str, optional
        x-axis variable. The default is 'season'.
    legx : str, optional
        x-axis label. The default is 'season'.
    vary : str, optional
        y-axis variable. The default is 'cadence'.
    legy : str, optional
        y-axis label. The default is 'cadence [day]'.
    plot_mean : bool, optional
        To superimpose the <y>. The default is True.

    Returns
    -------
    None.

    """

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
    """
    plots of vary vs varx

    Parameters
    ----------
    sel : pandas df
        Data to plot.
    varx : str
        x-axis variable.
    legx : str
        x-axis legend.
    vary : str
        y-axis variable.
    legy : str
        y-axis legend.

    Returns
    -------
    None.

    """

    fields = sel['field'].unique()
    dbName = sel['dbName'].unique()[0]

    bands = 'ugrizy'
    for field in fields:
        idx = sel['field'] == field
        selb = sel[idx]
        figtitle = f'{dbName} - {field}'
        plot_var_mean(selb, figtitle=figtitle, varx=varx,
                      legx=legx, vary=vary, legy=legy)
        for b in bands:
            vvary = f'{vary}_{b}'
            figtitle = f'{dbName} - {field} - {b} band'
            plot_var_mean(selb, figtitle=figtitle, varx=varx,
                          legx=legx, vary=vvary, legy=legy)


def print_pixel_info(sel, healpixID):
    """
    Function to grab pixel info

    Parameters
    ----------
    sel : pandas df
        Data to process.
    healpixID : int
        healpix ID.

    Returns
    -------
    None.

    """

    idxb = sel['healpixID'] == healpixID
    selnc = sel[idxb]
    print(selnc[['healpixID', 'pixRA', 'pixDec', 'nvisits', 'cadence', 'season']])


def load_data(dbDir, dbName, fields, fieldType='DD'):
    """
    Function to load the data

    Parameters
    ----------
    dbDir : str
        data main directory.
    dbName : str
        OS to analyze.
    fields : list(str)
        list of fields to process.
    fieldType : str, optional
        type of field (DD/WFD). The default is 'DD'.

    Returns
    -------
    df : pandas df
        output data.

    """

    theDir = '{}/{}'.format(dbDir, dbName)
    prefix = '{}_pixels_{}'.format(fieldType, dbName)

    # grab the list of data
    list_data = []
    if fieldType == 'DD':
        # loop on ddfs
        for field in fields:
            fName = '{}/{}_{}*.hdf5'.format(theDir, prefix, field)
            fis = glob.glob(fName)
            if len(fis) == 0:
                print('data not found', fName)
            list_data += fis
    if fieldType == 'WFD':
        fName = '{}/{}_*.hdf5'.format(theDir, prefix)
        fis = glob.glob(fName)
        if len(fis) == 0:
            print('data not found', fName)
        list_data += fis

    # load the data
    df = pd.DataFrame()
    for fi in list_data:
        dd = pd.read_hdf(fi)
        df = pd.concat((df, dd))

    return df


parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbName', type=str, default='test_newb',
                  help='dbName to process [%default]')
parser.add_option('--dbDir', type=str, default='../test_metric',
                  help='dbDir of the OS to process [%default]')
parser.add_option('--nside', type=int, default=128,
                  help='healpix nside parameter [%default]')
parser.add_option('--plots', type=str,
                  default='cadence_season,nvisits_season,cadence_dist,nvisits_dist,mollview_cadence,mollview_nvisits',
                  help='plots to show [%default]')
parser.add_option('--mollview_seasons', type=str,
                  default='1-5',
                  help='plots to show [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='plots to show [%default]')
parser.add_option('--fieldType', type=str,
                  default='DD',
                  help='type of field to process (DD/WFD) [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='time scale for the plots (year/season) [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
nside = opts.nside
plots = opts.plots.split(',')
mollview_seasons = opts.mollview_seasons
fields = opts.fields.split(',')
fieldType = opts.fieldType
timescale = opts.timescale

if '-' in mollview_seasons:
    cad_brk = mollview_seasons.split('-')
    moll_seasons = list(range(int(cad_brk[0]), int(cad_brk[1])+1))
else:
    moll_seasons = list(map(int, mollview_seasons.split(',')))

print('seasons moll', moll_seasons)

# fName = '{}/{}.hdf5'.format(dbDir, dbName)

df = load_data(dbDir, dbName, fields, fieldType=fieldType)

# df = pd.read_hdf(fName)
df['dbName'] = dbName
print(df.columns)
# print(test)

idx = df[timescale] > 0
idx &= df[timescale] < 11
idx &= df['cadence'] > 0.
sel = df[idx]


if 'cadence_season' in plots:
    multiplot_season(sel, varx=timescale, legx=timescale,
                     vary='cadence', legy='cadence [day]')
if 'nvisits_season' in plots:
    multiplot_season(sel, varx=timescale, legx=timescale,
                     vary='nvisits', legy='N$_{visits}$')
if 'cadence_dist' in plots:
    multiplot_dist(sel)
if 'nvisits_dist' in plots:
    multiplot_dist(sel, yvar='nvisits', yleg=r'N$_{visits}$')
if 'mollview_cadence' in plots:
    plotMollview_seasons(nside, sel, dbName,
                         yvar='cadence', yleg='cadence [day]',
                         op=np.mean, seasons=moll_seasons)
if 'mollview_nvisits' in plots:
    plotMollview_seasons(nside, sel, dbName,
                         yvar='nvisits', yleg='N$_{visits}$',
                         op=None, seasons=moll_seasons)

"""
print_pixel_info(sel, 109384)
print_pixel_info(sel, 109031)
"""

plt.show()
