#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:19:30 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import glob
from sn_plotter_metrics.plot4metric import plot_per_bin
import matplotlib.pyplot as plt
import numpy as np


def load_data(dataDir, dbName):
    """
    Function to load the data

    Parameters
    ----------
    dataDir : str
        Data directory.
    dbName : str
        db Name.

    Returns
    -------
    dft : pandas df
        Loaded data.

    """

    fName = '{}/*{}*_for_fit.hdf5'.format(dataDir, dbName)

    fis = glob.glob(fName)

    dft = pd.DataFrame()
    for fi in fis:
        df_ = pd.read_hdf(fi)
        dft = pd.concat((dft, df_))

    return dft


def plot_sigma_mu(data, config):
    """
    Function to plot sigma_mu stat

    Parameters
    ----------
    data : pandas df
        Data to plot.
    config : pandas df
        config file.

    Returns
    -------
    None.

    """

    print(data.columns)
    # yearly
    plot_sigma_mu_yearly(data, config)

    # all years
    plot_sigma_full_survey(data, config)


def plot_sigma_mu_yearly(data, config):
    """
    Function to plot sigma_mu stat yearly

    Parameters
    ----------
    data : pandas df
        Data to plot.
    config : pandas df
        config file.

    Returns
    -------
    None.

    """
    years = data['year'].unique()

    for year in years:
        idx = data['year'] == year
        sel_y = data[idx]
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.subplots_adjust(right=0.75)
        fig.suptitle('year {}'.format(year))
        dbNames = sel_y['dbName'].unique()
        for dbName in dbNames:
            idxb = sel_y['dbName'] == dbName
            idxb &= sel_y['sigma_mu'] <= 0.5
            selb = sel_y[idxb]
            idxcc = config['dbName'] == dbName
            selp = config[idxcc]
            dbNameb = selp['dbName_plot'].unique()[0]
            marker = selp['marker'].unique()[0]
            color = selp['color'].unique()[0]
            ls = selp['ls'].unique()[0]
            plot_per_bin(ax, selb, 'sigma_mu', 'nsn',
                         smoothIt=True, ls=ls, color=color,
                         marker=marker, label=dbNameb,
                         bins=np.arange(0., 0.52, 0.01),
                         xmin=0., xmax=0.5, ymin=0., sumIt=True, norm=False)
        ax.grid(visible=True)
        ax.legend(bbox_to_anchor=(1., 0.5), ncol=1, frameon=False, fontsize=15)

        ymin, ymax = ax.get_ylim()
        sigmin = 0.12
        ax.plot([sigmin, sigmin], [ymin, ymax], color='k', linestyle='dashed')
        ax.set_xlabel('$\sigma_{\mu}$')
        ax.set_ylabel('$\Sigma N_{SN}^{DDF}$')


def plot_sigma_full_survey(data, config):
    """
    Function to plot sigma_mu stat - full survey

    Parameters
    ----------
    data : pandas df
        Data to plot.
    config : pandas df
        config file.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.subplots_adjust(right=0.75)

    dbNames = data['dbName'].unique()
    for dbName in dbNames:
        idxb = data['dbName'] == dbName
        idxb &= data['sigma_mu'] <= 0.5
        selb = data[idxb]
        idxcc = config['dbName'] == dbName
        selp = config[idxcc]
        dbNameb = selp['dbName_plot'].unique()[0]
        marker = selp['marker'].unique()[0]
        color = selp['color'].unique()[0]
        ls = selp['ls'].unique()[0]
        plot_per_bin(ax, selb, 'sigma_mu', 'nsn',
                     smoothIt=True, ls=ls, color=color,
                     marker=marker, label=dbNameb,
                     bins=np.arange(0., 0.52, 0.01),
                     xmin=0., xmax=0.5, ymin=0., sumIt=True, norm=False)

    ax.grid(visible=True)
    ax.legend(bbox_to_anchor=(1., 0.5), ncol=1, frameon=False, fontsize=15)

    ymin, ymax = ax.get_ylim()
    sigmin = 0.12
    ax.plot([sigmin, sigmin], [ymin, ymax], color='k', linestyle='dashed')
    ax.set_xlabel('$\sigma_{\mu}$')
    ax.set_ylabel('$\Sigma N_{SN}^{DDF}$')
    plt.show()


parser = OptionParser(
    description='Script to plot the surveys used to estimate cosmo params')

parser.add_option('--dataDir', type=str,
                  default='../test_durvey',
                  help='Data dir [%default]')
parser.add_option('--config', type=str,
                  default='config_ana_selplot.csv',
                  help='config file [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='data type [%default]')
opts, args = parser.parse_args()

dataDir = opts.dataDir
config = opts.config
fields = opts.fields.split(',')

# load config
df_config = pd.read_csv(config, comment='#')

# load the data
sn_cosmo = pd.DataFrame()
for i, row in df_config.iterrows():
    dbName = row['dbName']
    df_ = load_data(dataDir, dbName)
    print(dbName, len(df_))
    if len(df_) > 0:
        df_['dbName'] = dbName
        sn_cosmo = pd.concat((sn_cosmo, df_))


# select the fields
idx = sn_cosmo['field'].isin(fields)

plot_sigma_mu(sn_cosmo[idx], df_config)
