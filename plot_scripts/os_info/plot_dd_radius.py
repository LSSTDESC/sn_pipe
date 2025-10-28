#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:03:39 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import numpy as np
from sn_plotter_metrics import plt


def plot_radius(data, df_config, year=1, field='COSMOS'):
    """
    Function to plot nvisits and cadence vs radius

    Parameters
    ----------
    data : pandas df
        Data to plot.
    df_config : pandas df
        config for the plot.
    year : int, optional
        year. The default is 1.
    field : str, optional
        Field. The default is 'COSMOS'.

    Returns
    -------
    None.

    """

    idx = data['year'] == year
    idx &= data['field'] == field

    sel = data[idx]

    dbNames = sel['dbName'].unique()

    fig, ax = plt.subplots(nrows=2, figsize=(14, 9))
    tit = '{} - year {}'.format(field, year)
    fig.suptitle(tit)
    fig.subplots_adjust(hspace=0.0, right=0.73)
    for i, row in df_config.iterrows():
        family = row['dbName_plot']
        marker = row['marker']
        color = row['color']
        ls = row['ls']
        idxb = sel['dbName'] == row['dbName']
        selb = sel[idxb]

        # vv = np.array(np.cumsum(selb['nvisits']))
        # vv = np.array(selb['nvisits'])
        mean_radius = selb['dist_center'].diff()

        area = np.pi*selb['dist_center']**2
        areas = area.diff()
        areas = areas.fillna(np.pi*0.375**2)

        ax[0].errorbar(selb['dist_center'], selb['nvisits_mean'], yerr=selb['nvisits_std'],
                       label=family, marker=marker, color=color, ls=ls, mfc='None')
        idx = selb['dist_center'] <= 1.5
        cad_low_dist = selb[idx]['cadence_mean'].mean()
        ax[1].errorbar(selb['dist_center'], selb['cadence_mean']-cad_low_dist, yerr=selb['cadence_std'],
                       label=family, marker=marker, color=color, ls=ls, mfc='None')

    ax[1].legend(bbox_to_anchor=(1., 1.5), ncol=1, frameon=False, fontsize=15)

    for i in range(2):
        ymin, ymax = ax[i].get_ylim()
        ax[i].plot([1.75]*2, [ymin, ymax], linestyle='dashed', color='r')
        ax[i].plot([2.0]*2, [ymin, ymax], linestyle='dashed', color='r')
        ax[i].set_ylim([ymin, ymax])
        ax[i].grid()

    ax[1].set_xlabel(r'distance w.r.t center [deg]')
    ax[0].set_ylabel(r'<$N_{visits}$>')
    ax[1].set_ylabel(r'<cadence> [day]')
    ax[0].get_xaxis().set_visible(False)


parser = OptionParser(description='Script to plot pixel level OS radius')

parser.add_option('--fName', type=str, default='data_radius.hdf5',
                  help='file name to process [%default]')
parser.add_option('--config', type=str, default='config_ana_selplot_part2.csv',
                  help='config file [%default]')

opts, args = parser.parse_args()

fName = opts.fName
configFile = opts.config

df = pd.read_hdf(fName)
df_config = pd.read_csv(configFile, comment='#')
print(df['dbName'].unique())
plot_radius(df, df_config)
plot_radius(df, df_config, year=2)
plot_radius(df, df_config, year=3)
plt.show()
