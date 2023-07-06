#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:08:41 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sn_plotter_cosmology.cosmoplot import Process_OS, cosmo_plot, cosmo_four
from optparse import OptionParser


def load_data(dbDir, config):
    """
    Method to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    config : pandas df
        config (dbName+OS params).

    Returns
    -------
    df : pandas df
        output data.

    """

    df = pd.DataFrame()

    for i, row in config.iterrows():
        fi = '{}/cosmo_{}.hdf5'.format(dbDir, row['dbName'])
        dfa = pd.read_hdf(fi)
        df = pd.concat((df, dfa))

    return df


def plot_allOS(resdf, config, prior='prior', vary='MoM', legy='$MoM$'):

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(right=0.8)
    dd = dict(zip(['noprior', 'prior'], ['no prior', 'with prior']))
    fig.suptitle(dd[prior])

    for i, row in config.iterrows():
        idx = resdf['dbName_DD'] == row['dbName']
        idx &= resdf['prior'] == prior
        sel = resdf[idx]
        cosmo_plot(sel, vary=vary, legy=legy, ax=ax, ls=row['ls'],
                   marker=row['marker'], color=row['color'], leg=row['dbName'])

    ax.grid()
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.15, 0.7),
              ncol=1, fontsize=12, frameon=False)


parser = OptionParser(description='Script to analyze SN prod')

parser.add_option('--dbDir', type=str, default='../cosmo_fit',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='input/DESC_cohesive_strategy/config_ana.csv',
                  help='OS name[%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList


config = pd.read_csv(dbList, comment='#')

data = load_data(dbDir, config)
# data['dbName_DD'] = 'DDF_Univ_WZ'
grpCol = ['season', 'dbName_DD', 'prior']
# resdf = process_OS(data, grpCol)
resdf = Process_OS(data, grpCol).res

print(resdf, config)

vvars = ['MoM', 'sigma_w']
leg = dict(zip(vvars, [r'$MoM$', r'$\sigma_w$[%]']))
for vary in vvars:
    for prior in ['prior', 'noprior']:
        plot_allOS(resdf, config, vary=vary, legy=leg[vary], prior=prior)
# ax.legend()
# idx = resdf['prior'] == 'noprior'
# cosmo_plot(resdf[idx])

# cosmo_four(resdf)
plt.show()
