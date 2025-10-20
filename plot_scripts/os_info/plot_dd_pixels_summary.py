#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:09:39 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import glob
import healpy as hp
# import matplotlib.pyplot as plt
from sn_plotter_analysis import plt


def load_data(dbDir, config, field):
    """
    Function to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    config : str
        config file.
    field : str
        Field to show.

    Returns
    -------
    df : pandas df
        Data to plot.

    """

    df_config = pd.read_csv(config, comment='#')

    df = pd.DataFrame()
    for i, row in df_config.iterrows():
        dbName = row['dbName']
        fName = '{}/{}/*{}*.hdf5'.format(dbDir, dbName, field)
        fi = glob.glob(fName)[0]
        ff = pd.read_hdf(fi)
        ff['dbName'] = dbName
        ff['dbName_plot'] = row['dbName_plot']
        ddi = 'unknown'
        ditype = 'unknown'
        if 'dither' in dbName:
            tt = dbName.split('_')
            ddi = tt[tt.index('dither')+1]
            ditype = 'between nights'
            if 'all' in dbName:
                ditype = 'nightly'

        ff['dither'] = ddi
        ff['dither_type'] = ditype

        df = pd.concat((df, ff))

    return df


def get_info_pixels(grp, pixArea):
    """
    Method to get pixel Infos

    Parameters
    ----------
    grp : pandas df
        Data to process.
    pixArea : float
        pixel area.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    idx = grp['cadence'] > 0.
    grpb = grp[idx]
    npixels = len(grpb['healpixID'].unique())
    area = npixels*pixArea

    return pd.DataFrame({'area': [area]})


def plot_timescale(df, varx='dbName_plot', legx='',
                   vary='area', legy='area [deg$^2$]',
                   timescale='year', figtitle=''):
    """
    Function to make plot for per varx/timescale

    Parameters
    ----------
    df : pandas df
        Data to process.
    varx : str, optional
        x-axis variable to plot. The default is 'dbName_plot'.
    legx : str, optional
        x-axis label. The default is ''.
    vary : str, optional
        y-axis variable to plot. The default is 'area'.
    legy : str, optional
        y-axis label. The default is 'area [deg$^2$]'.
    timescale : str, optional
        Timescale (year/season). The default is 'year'.
    figtitle : str, optional
        figure title. The default is ''.

    Returns
    -------
    None.

    """

    seasons = range(1, 11)
    mms = ['o', 's', 'H', '^', 'v']*2
    ls = ['solid']*5+['dotted']*5
    markers = dict(zip(seasons, mms))
    lns = dict(zip(seasons, ls))

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(figtitle)

    ida = df[timescale] > 0
    df = df[ida]
    timescales = df[timescale].unique()

    for tt in timescales:
        idx = df[timescale] == tt
        sel = df[idx]
        sel = sel.sort_values(by=[varx])
        ax.plot(sel[varx], sel[vary], marker=markers[tt],
                linestyle=lns[tt], mfc='None',
                label='Y{}'.format(int(tt)))

    if legx != '':
        ax.set_xlabel(r'{}'.format(legx))
    if legy != '':
        ax.set_ylabel(r'{}'.format(legy))

    ax.legend()
    ax.grid(visible=True)
    ax.tick_params(axis='x', labelrotation=20., labelsize=12, labelright=True)


def plot_dither(df, varx='dither', legx='dither',
                vary='area', legy='area [deg$^2$]', figtitle=''):
    """
    Function to plot dither stuff

    Parameters
    ----------
    df : pandas df
     Data to process.
     varx : str, optional
         x-axis variable to plot. The default is 'dither'.
    legx : str, optional
     x-axis label. The default is 'dither'.
    vary : str, optional
     y-axis variable to plot. The default is 'area'.
    legy : str, optional
     y-axis label. The default is 'area [deg$^2$]'.
     figtitle : str, optional
     figure title. The default is ''.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(figtitle)

    idx = df[varx] != 'unknown'

    sel = df[idx]

    di_types = sel['dither_type'].unique()

    marks = dict(zip([0, 1], ['o', 's']))

    for i, val in enumerate(di_types):
        idxb = sel['dither_type'] == val
        selb = sel[idxb]
        ax.plot(selb[varx], selb[vary], 'k{}'.format(marks[i]), label=val)

    ax.set_xlabel(r'{}'.format(legx))
    ax.set_ylabel(r'{}'.format(legy))

    ax.grid(visible=True)
    ax.legend()


parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--config', type=str, default='config_ana_selplot_part1.csv',
                  help='config file [%default]')
parser.add_option('--dbDir', type=str, default='../dd_pixels',
                  help='data dir [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to show [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale for the plots (year/season) [%default]')
parser.add_option('--nside', type=int,
                  default='128',
                  help='nside healpix parameter [%default]')

opts, args = parser.parse_args()

config = opts.config
dbDir = opts.dbDir
fields = opts.fields.split(',')
timescale = opts.timescale
nside = opts.nside

pixArea = hp.nside2pixarea(nside, degrees=True)

for field in fields:
    # load the data
    df = load_data(dbDir, config, field)
    dfb = df.groupby(['dbName', 'dbName_plot', 'dither', 'dither_type', timescale]).apply(
        lambda x: get_info_pixels(x, pixArea=pixArea), include_groups=False).reset_index()
    print(dfb)
    plot_timescale(dfb, figtitle=field)

    """
    dfc = dfb.groupby(['dbName', 'dbName_plot', 'dither', 'dither_type'])[
        'area'].mean().reset_index()
    """
    dfc = df.groupby(['dbName', 'dbName_plot', 'dither', 'dither_type']).apply(
        lambda x: get_info_pixels(x, pixArea=pixArea), include_groups=False).reset_index()
    print(dfc)

    plot_dither(dfc, figtitle=field)

plt.show()
