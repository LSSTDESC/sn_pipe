#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 08:37:32 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sn_plotter_analysis.sn_analyser_wdf import plotMollview


def load_data(dbDir, dbName):
    """
    Function to load data

    Parameters
    ----------
    dbDir : str
        File location dir.
    dbName : str
        db name to process.

    Returns
    -------
    dft : pandas df
        output data.

    """

    dirFiles = '{}/{}'.format(dbDir, dbName)

    fis = glob.glob('{}/*.hdf5'.format(dirFiles))

    dft = pd.DataFrame()
    for fi in fis:
        dfa = pd.read_hdf(fi)
        dft = pd.concat((dft, dfa))

    dft['dbName'] = dbName
    return dft


def plot_nvisits_histo(data):
    """
    Plot nvisits (10 years) histogram

    Parameters
    ----------
    data : pandas df
        Data to plot.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(12, 8))

    idx = data['season'] == -1

    # idx &= data['nvisits'] <= 1000
    """
    idx &= data['nvisits'] >= 700
    """
    seld = data[idx]

    dbNames = seld['dbName'].unique()

    for dbName in dbNames:
        idxb = seld['dbName'] == dbName
        selc = seld[idxb]
        ax.hist(selc['nvisits'], histtype='step', bins=50)

    ax.grid(visible=True)

    ax.set_xlabel('N$_{visits}$')
    ax.set_ylabel('Number of Entries')


def plot_mollweid(data, varname='nvisits'):
    """
    Mollweide plot

    Parameters
    ----------
    data : pandas df
        Data to plot.
    varname : str, optional
        column to plot. The default is 'nvisits'.

    Returns
    -------
    None.

    """

    idx = data['season'] == -1

    # idx &= data['nvisits'] <= 1000
    """
    idx &= data['nvisits'] >= 700
    """
    seld = data[idx]

    dbNames = seld['dbName'].unique()

    for dbName in dbNames:
        idxb = seld['dbName'] == dbName
        selc = seld[idxb]
        xmin, xmax = selc[varname].min(), selc[varname].max()
        plotMollview(selc, varname, dbName, xmin, xmax, nside=nside)


def plot_compare_pixels(datam, ref='baseline_v4.3.1_10yrs', colName='dbName'):

    idxa = datam['season'] == -1
    data = datam[idxa]

    idx = data[colName] == ref

    ref_df = pd.DataFrame(data[idx])

    oth_df = pd.DataFrame(data[~idx])

    oth_df = oth_df.merge(
        ref_df, left_on=['healpixID'], right_on=['healpixID'])

    oth_df['delta_nvisits'] = 100.*(
        oth_df['nvisits_x']-oth_df['nvisits_y'])/oth_df['nvisits_y']

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(right=0.75)

    ii = oth_df['nvisits_x'] > 700.
    ii &= oth_df['nvisits_x'] <= 1000

    oth_df = oth_df[ii]
    dbNames = oth_df['{}_x'.format(colName)].unique()

    for dbName in dbNames:
        idx = oth_df['dbName_x'] == dbName
        sel = oth_df[idx]

        print(dbName, sel['delta_nvisits'].median(),
              sel['delta_nvisits'].std())
        idc = conf_df['dbName'] == dbName
        selp = conf_df[idc]
        ls = selp['ls'].values[0]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        dbNameb = selp['dbName_plot'].values[0]
        ax.hist(sel['delta_nvisits'],
                histtype='step', bins=20, color=color, linestyle=ls, label=dbNameb)
    ax.grid(visible=True)
    ax.set_xlabel(r'$\frac{\Delta N_{visits}}{N_{visits}}$[%]')
    ax.set_ylabel('Number of Entries')
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.20, 0.7),
              ncol=1, fontsize=12, frameon=False)


parser = OptionParser(description='Script to analyze SN - DDF after selection')

parser.add_option('--dbDir', type=str,
                  default='../wfd_pixels',
                  help='OS location dir [%default]')
parser.add_option('--config', type=str,
                  default='config_ana_selplot.csv',
                  help='configuration for the plots [%default]')
parser.add_option('--nside', type=int,
                  default=64,
                  help='nside healpix parameter [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
config = opts.config
nside = opts.nside

# load config file
conf_df = pd.read_csv(config, comment='#')

# load the data
dbNames = conf_df['dbName'].unique()

data = pd.DataFrame()
for i, row in conf_df.iterrows():
    dbName = row['dbName']
    dfa = load_data(dbDir, dbName)
    data = pd.concat((data, dfa))

print(data)

# plot_nvisits_histo(data)

# plot_mollweid(data)

plot_compare_pixels(data)

plt.show()
