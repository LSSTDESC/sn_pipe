#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:08:52 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import matplotlib.pyplot as plt
from sn_plotter_os_info.os_info_util import plot_summary


def plot_nvisits_all(data, dfconfig, bands='ugrizy'):

    for b in bands:
        plot_nvisits(data, df_config, b)


def plot_nvisits(data, df_config, b, field='DD:ECDFS'):

    plot_summary(data, field=field,
                 varx='year', labx='year',
                 vary='Nvisits_{}'.format(b), laby='$\Sigma N_{visits}$ - band',
                 figtit=field,
                 df_config=df_config)


parser = OptionParser(description='Script to plot calib reqs from PZ and WL')

parser.add_option('--config', type=str,
                  default='config_ana_selplot.csv',
                  help='OS DD list[%default]')
parser.add_option("--dirFile", type="str",
                  default='../nvisits_m5',
                  help="file directory [%default]")
opts, args = parser.parse_args()

dirFile = opts.dirFile
config = opts.config

# load config
df_config = pd.read_csv(config, comment='#')

# load the data
data = pd.DataFrame()
for i, row in df_config.iterrows():
    df = pd.read_hdf('{}/{}.hdf5'.format(dirFile, row['dbName']))
    data = pd.concat((data, df))

print('there man', data)

plot_nvisits_all(data, df_config)

plt.show()
