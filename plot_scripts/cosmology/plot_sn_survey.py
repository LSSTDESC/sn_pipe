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
import time
from sn_analysis.sn_calc_plot import bin_it
from sn_tools.sn_utils import multiproc
from sn_plotter_analysis.sn_plot import plot_ddf_year


def plot_ddf(df, pp):

    print('alllll', df.columns)

    plots = pp['plots'].split(',')
    ud_fields = pp['ud_fields'].split(',')
    dd_fields = pp['dd_fields'].split(',')

    # read config file
    conf_df = pd.read_csv(pp['config'], comment='#')
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


parser = OptionParser('script to plot LSST SN surveys')

parser.add_option('--dataDir', type=str,
                  default='../sn_summary_surveys',
                  help='data directory [%default]')
parser.add_option('--surveyList', type=str, default='list_surveys_plot.csv',
                  help='OS for DD [%default]')
parser.add_option('--plots', type=str,
                  default='nsn_all,nsn_ud',
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
parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS DD list[%default]')

opts, args = parser.parse_args()

pp = vars(opts)


# load data

dbList = pd.read_csv(pp['surveyList'], comment='#')

df = pd.DataFrame()

for i, row in dbList.iterrows():
    fName = '{}/{}/sn_survey.hdf5'.format(pp['dataDir'], row['dbName'])
    df_ = pd.read_hdf(fName)
    print('allo', df_.columns)
    df = pd.concat((df, df_))

plot_ddf(df, pp)

plt.show()
