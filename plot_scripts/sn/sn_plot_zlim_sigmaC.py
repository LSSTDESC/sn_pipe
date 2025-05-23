#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:33:01 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser
import matplotlib.pyplot as plt
from sn_plotter_analysis import plt


def load_data(dbDir, df_list):

    res = pd.DataFrame()

    for i, row in df_list.iterrows():
        df_ = pd.read_hdf('{}/{}.hdf5'.format(dbDir, row['dbName']))
        res = pd.concat((res, df_))

    return res


def plot_zlim(data, config, field='COSMOS'):

    idx = data['field'] == field

    seldata = data[idx]

    dbNames = seldata['dbName'].unique()

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(right=0.75)
    fig.suptitle(field, fontweight='bold')
    for dbName in dbNames:
        idx = seldata['dbName'] == dbName
        selplot = seldata[idx]
        selplotb = selplot.groupby(['season'])['zlim'].mean().reset_index()
        selplotb = selplot.groupby(['season'], as_index=False).agg(
            {'zlim': ['mean', 'std']})

        ida = config['dbName'] == dbName
        color = config[ida]['color'].values[0]
        ls = config[ida]['ls'].values[0]
        marker = config[ida]['marker'].values[0]
        dbNameb = config[ida]['dbName_plot'].values[0]
        ax.plot(selplotb['season'], selplotb['zlim']['mean'], linestyle=ls,
                color=color, marker=marker, mfc='None',
                label=dbNameb, lw=3, markersize=10)
        """
        ax.errorbar(selplotb['season'], selplotb['zlim']['mean'],
                    yerr=selplotb['zlim']['std'], linestyle=ls,
                    color=color, marker=marker, mfc='None', label=dbNameb)
        """
    ax.grid(visible=True)
    ax.legend(bbox_to_anchor=(1., 0.8), ncol=1, frameon=False, fontsize=15)
    ax.set_xlabel(r'season')
    ax.set_ylabel(r'$z_{lim}^{\sigma_{C} \leq 0.04}$')
    plt.tight_layout()


parser = OptionParser(description='Script to plot zlim for sigmaC<=0.04')

parser.add_option('--dbDir', type=str,
                  default='../sn_zlim_sigmaC',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='config_ana_selplot.csv',
                  help='OS name [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList


# load dd list

df_list = pd.read_csv(dbList, comment='#')

# load data

data = load_data(dbDir, df_list)

print(data)

plot_zlim(data, df_list, field='COSMOS')
plot_zlim(data, df_list, field='XMM-LSS')

plt.show()
