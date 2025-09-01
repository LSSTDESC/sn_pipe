#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 09:55:57 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import glob
import pandas as pd
from sn_cosmology.cosmo_tools import get_seasons
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sn_analysis.sn_calc_plot import bin_it
from sn_tools.sn_utils import multiproc
from sn_plotter_analysis.sn_plot import plot_ddf_year
from sn_plotter_analysis.sn_analyser_tools import print_nsn_latex


def plot_ddf(df, pp, conf_df):
    """
    Function to plot the data

    Parameters
    ----------
    df : pandas df
        data to plot.
    pp : dict
        parameters.
    conf_df: pandas df.
       config data (linestyles, colors...)

    Returns
    -------
    None.

    """

    plots = pp['plots'].split(',')
    ud_fields = pp['ud_fields'].split(',')
    dd_fields = pp['dd_fields'].split(',')
    wfd_fields = pp['wfd_fields'].split()

    # all fields
    if 'nsn_all' in plots:
        fields = ud_fields+dd_fields
        plot_ddf_year(df, conf_df,
                      cols=['year', 'dbName'],
                      fields=fields)

    # UD only
    if 'nsn_ud' in plots:

        fields = ud_fields
        plot_ddf_year(df, conf_df,
                      cols=['year', 'dbName'],
                      fields=fields)

    # WFD only
    if 'wfd' in plots:

        fields = wfd_fields
        plot_ddf_year(df, conf_df,
                      cols=['year', 'dbName'],
                      fields=fields)


def plot_effi(df_effi, config, fields=['COSMOS', 'WFD'], years=[1, 2]):
    """
    Function to plot efficiencies vs z

    Parameters
    ----------
    df_effi : pandas df
        Data to plot.
    config: pandas df.
      config for the plot (colors, linestyles, ...)
    fields : list(str), optional
        List of fields to consider. The default is ['COSMOS', 'WFD'].
    years : list(int), optional
        List of years to consider. The default is [1, 2].

    Returns
    -------
    None.

    """

    dbNames = df_effi['dbName'].unique()
    xlab = 'z'
    ylab = 'efficiency'
    for field in fields:
        for year in years:
            effi = df_effi[(df_effi['field'] == field)
                           & (df_effi['year'] == year)]
            fig, ax = plt.subplots(figsize=(15, 8))
            fig.subplots_adjust(right=0.78)
            fig.suptitle('{} - year {}'.format(field, year))
            for dbName in dbNames:  # get config for plot
                idxb = config['dbName'] == dbName
                selconf = config[idxb]
                ls = selconf['ls'].values[0]
                color = selconf['color'].values[0]
                mark = selconf['marker'].values[0]
                name = selconf['dbName_plot'].values[0]
                effip = effi[effi['dbName'] == dbName]
                ax.errorbar(effip['z_fit'], effip['effi'],
                            yerr=effip['err_effi'], color=color,
                            marker=mark, linestyle=ls,
                            label=name, mfc='None', lw=2, ms=10)

                ax.grid(visible=True)
                ax.set_xlabel(r'{}'.format(xlab))
                ax.set_ylabel(r'{}'.format(ylab))
                # ax.set_xlim([0.9, 10.1])
                ax.legend(loc='center left', bbox_to_anchor=(
                    1, 0.5), ncol=1, fontsize=14, frameon=False)


parser = OptionParser('script to plot LSST SN surveys')

parser.add_option('--dataDir', type=str,
                  default='../sn_summary_surveys',
                  help='data directory [%default]')
parser.add_option('--surveyList', type=str, default='list_surveys_plot.csv',
                  help='OS for DD [%default]')
parser.add_option('--plots', type=str,
                  default='nsn_all,nsn_ud,wfd',
                  help='plots to draw [%default]')
parser.add_option('--print_nsn', type=int,
                  default=0,
                  help='to print nsn as a latex table [%default]')
parser.add_option('--ud_fields', type=str,
                  default='COSMOS,XMM-LSS',
                  help='UD fields to consider [%default]')
parser.add_option('--dd_fields', type=str,
                  default='CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='DD fields to consider [%default]')
parser.add_option('--wfd_fields', type=str,
                  default='WFD',
                  help='DD fields to consider [%default]')
parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS DD list [%default]')
parser.add_option('--genplot', type=str,
                  default='survey_spectro,survey_all,effi',
                  help='OS DD list [%default]')
parser.add_option('--effi_fields', type=str,
                  default='COSMOS,WFD',
                  help='list of fields for effi plots [%default]')
parser.add_option('--effi_years', type=str,
                  default='1,3',
                  help='list of years for effi plots [%default]')

opts, args = parser.parse_args()

pp = vars(opts)


# load data

dbList = pd.read_csv(pp['surveyList'], comment='#')

df = pd.DataFrame()
df_all = pd.DataFrame()
df_effi = pd.DataFrame()

for i, row in dbList.iterrows():
    fName = '{}/{}/sn_survey.hdf5'.format(pp['dataDir'], row['dbName'])
    df_ = pd.read_hdf(fName)
    df = pd.concat((df, df_))
    fName_all = '{}/{}/sn_survey_all.hdf5'.format(pp['dataDir'], row['dbName'])
    if os.path.exists(fName_all):
        df__ = pd.read_hdf(fName_all)
        df_all = pd.concat((df_all, df__))
    effiName = '{}/{}/effi_spectro.hdf5'.format(pp['dataDir'], row['dbName'])
    if os.path.exists(effiName):
        df___ = pd.read_hdf(effiName)
        df_effi = pd.concat((df_effi, df___))


# read config file
conf_df = pd.read_csv(pp['config'], comment='#')

tp = pp['genplot'].split(',')

if 'survey_spectro' in tp:
    plot_ddf(df, pp, conf_df)

    if pp['print_nsn']:
        print_nsn_latex(df)

if 'survey_all' in tp:
    if len(df_all) > 0:
        plot_ddf(df_all, pp, conf_df)
        if pp['print_nsn']:
            print_nsn_latex(df_all)

if 'effi' in tp:
    if len(df_effi) > 0:
        yyears = pp['effi_years'].split(',')
        yyears = list(map(int, yyears))
        plot_effi(df_effi, conf_df,
                  fields=pp['effi_fields'].split(','),
                  years=yyears)


plt.show()
