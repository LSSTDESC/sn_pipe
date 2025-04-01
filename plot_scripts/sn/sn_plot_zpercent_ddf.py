#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:15:32 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import numpy as np
from sn_analysis import plt


def plot_zper(data, config, timescale, figtit=''):
    """
    Function to plot zpercentiles vs timescale

    Parameters
    ----------
    data : pandas df
        Data to process.
    config : pandas df
        Plot config.
    timescale : str
        Time scale (year/season).
    figtit : str, optional
        Figure title. The default is ''.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(15, 8), nrows=2)
    fig.subplots_adjust(right=0.78, wspace=0., hspace=0.)
    fig.suptitle(figtit)

    fracs = [0.80, 0.90]
    labels = [1, 0]

    for i, val in enumerate(fracs):
        idx = np.abs(data['frac']-val) < 0.01
        sel = data[idx]
        plot_frac(sel, ax[i], config, timescale, label=labels[i])

    for i, val in enumerate(fracs):
        ax[i].grid(visible=True)
        ax[i].set_xlim([1, 10])
        ylab = '$z^{'+'{}'.format(np.round(val, 2))+'}$'
        ax[i].set_ylabel(ylab)

    ax[0].legend(loc='center left', bbox_to_anchor=(
        1, 0.), ncol=1, fontsize=14, frameon=False)
    ax[1].set_xlabel(r'year', fontweight='bold')
    ax[0].set_xticklabels([])


def plot_frac(data, ax, config, timescale, label=True):
    """
    Function to plot one of the (two) axis

    Parameters
    ----------
    data : pandas df
        Data to plot.
    ax : matplotlib axis
        Axis for the plot.
    config : pandas df
        Plot config.
    timescale : str
        Time scale (year/season).
    label : bool, optional
        plot label. The default is True.

    Returns
    -------
    None.

    """

    dbNames = data['dbName'].unique()

    for dbName in dbNames:
        idx = data['dbName'] == dbName
        sel_data = data[idx]

        # get config for plot
        idxb = config['dbName'] == dbName
        selconf = config[idxb]
        ls = selconf['ls'].values[0]
        color = selconf['color'].values[0]
        mark = selconf['marker'].values[0]
        name = selconf['dbName_plot'].values[0]
        if label:
            ax.plot(sel_data[timescale], sel_data['zlim'], color=color,
                    marker=mark, linestyle=ls, label=name, mfc='None', lw=2, ms=10)

        else:
            ax.plot(sel_data[timescale], sel_data['zlim'], color=color,
                    marker=mark, linestyle=ls, mfc='None', lw=2, ms=10)


parser = OptionParser(description='Script to plot zpercentiles for DDF')

parser.add_option('--fileName', type=str,
                  default='zpercentiles_ddf.hdf5',
                  help='data to plot [%default]')
parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS DD list[%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='data type [%default]')

opts, args = parser.parse_args()

fileName = opts.fileName
config = opts.config
timescale = opts.timescale
fields = opts.fields.split(',')
# read config file
conf_df = pd.read_csv(config, comment='#')

df = pd.read_hdf(fileName)

for field in fields:
    idx = df['field'] == field
    sel = df[idx]
    plot_zper(sel, conf_df, timescale, figtit=field)

plt.show()
