#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 11:01:44 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import matplotlib.pyplot as plt
from sn_plotter_analysis import plt
from optparse import OptionParser
import pandas as pd


def get_nvisits(grp):
    """
    Function to estimate the number of visits per season

    Parameters
    ----------
    grp : pandas df
        Data to use for the estimation of Nvisits.

    Returns
    -------
    df : pandas df
        original data plus nvisits_season column.

    """

    grp['nvisits_season'] = grp['nvisits_night']*grp['sl']/grp['cad']

    res = grp['nvisits_season'].sum()

    df = pd.DataFrame([res], columns=['nvisits'])

    return df


parser = OptionParser(description='Plot LSST DDF cohesive strategy results')

parser.add_option("--nvisits_lsst", type=float,
                  default=2.e6,
                  help="Total number of LSST visits (10 years) [%default]")
parser.add_option("--config", type=str,
                  default='config_lsst_ddf.csv',
                  help="List of files to process [%default]")
parser.add_option("--inputDir", type=str,
                  default='../lsst_ddf_cohesive',
                  help="input file dir[%default]")

opts, args = parser.parse_args()

nvisits_lsst = opts.nvisits_lsst
config = opts.config
inputDir = opts.inputDir

# load config file
df_conf = pd.read_csv(config)

# load files
res = pd.DataFrame()

for i, row in df_conf.iterrows():
    fName = '{}/{}.csv'.format(inputDir, row['fName'])
    df_ = pd.read_csv(fName)
    df_['survey_label'] = row['label']
    res = pd.concat((res, df_))

fig, ax = plt.subplots(figsize=(14, 9))
fig.suptitle('LSST DDF cohesive surveys', weight='bold')
surveys = res['survey_label'].unique()

for survey in surveys:
    idx = res['survey_label'] == survey
    idx &= res['zcomp'] >= 0.6
    sel = res[idx]
    resb = sel.groupby(['zcomp']).apply(
        lambda x: get_nvisits(x)).reset_index()
    resb['budget'] = 100. * resb['nvisits']/nvisits_lsst
    idxb = df_conf['label'] == survey
    selp = df_conf[idxb]
    marker = selp['marker'].values[0]
    color = selp['color'].values[0]
    ls = selp['ls'].values[0]
    ax.plot(resb['zcomp'], resb['budget'], linestyle=ls,
            marker=marker, color=color, label=survey, mfc='None')

ax.grid(visible=True)

ax.set_xlabel(r'$z_{comp}$')
ax.set_ylabel(r'DD budget [%]')
ax.set_xlim([0.6, 0.8])
ax.legend()
plt.show()
