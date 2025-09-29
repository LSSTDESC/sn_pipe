#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:16:35 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_analysis import plt
import os

from sn_plotter_analysis.sn_plot import plot_ddf_year
from sn_plotter_analysis.sn_analyser_tools import print_nsn_latex

parser = OptionParser(description='Script to plot SN - DDF after selection')

parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS DD list[%default]')
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
parser.add_option('--inputDir', type=str,
                  default='../sn_dd_airmass',
                  help='input dir for the file to draw [%default]')
parser.add_option('--fileName', type=str,
                  default='sn_summary_ddf.hdf5',
                  help='sn file name to draw [%default]')


opts, args = parser.parse_args()


config = opts.config
plots = opts.plots.split(',')
print_nsn = opts.print_nsn
ud_fields = opts.ud_fields.split(',')
dd_fields = opts.dd_fields.split(',')
inputDir = opts.inputDir
fileName = opts.fileName


# read config file
conf_df = pd.read_csv(config, comment='#')

# load the data

dbNames = conf_df['dbName'].unique()

df_nsn = pd.DataFrame()
for dbName in dbNames:
    fName = '{}/{}/{}'.format(inputDir, dbName, fileName)

    # process data
    if not os.path.isfile(fName):
        print('File not found!', fName)
        print('Please consider processing the data!')
    else:
        df_ = pd.read_hdf(fName)
        df_nsn = pd.concat((df_nsn, df_))

if print_nsn:
    print_nsn_latex(df_nsn)

# all fields
if 'nsn_all' in plots:
    fields = ud_fields+dd_fields
    plot_ddf_year(df_nsn, conf_df,
                  cols=['year', 'dbName'],
                  fields=fields)

# UD only
if 'nsn_ud' in plots:

    fields = ud_fields
    plot_ddf_year(df_nsn, conf_df,
                  cols=['year', 'dbName'],
                  fields=fields)

# DD only
if 'nsn_dd' in plots:

    fields = dd_fields
    plot_ddf_year(df_nsn, conf_df,
                  cols=['year', 'dbName'],
                  fields=fields)


plt.show()
