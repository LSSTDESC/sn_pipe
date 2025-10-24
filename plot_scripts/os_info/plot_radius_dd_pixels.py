#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 15:19:40 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from sn_plotter_metrics.utils import get_dist


def load_db(dbDir):

    search_path = '{}/*.hdf5'.format(dbDir)
    print('path', search_path)
    fis = glob.glob(search_path)

    df = pd.DataFrame()

    for fi in fis:
        tt = pd.read_hdf(fi)
        df = pd.concat((df, tt))

    return df


def rebin(df_dist, distval='dist_center', yvar='nvisits'):

    # rebin to have a "better" plot
    xmin, xmax = df_dist[distval].min(), df_dist[distval].max()
    bins = np.linspace(xmin-1.e-6, xmax, 12)
    # bins = np.arange(0.1, 2.22, 0.22)
    group = df_dist.groupby(pd.cut(df_dist[distval], bins), observed=False)
    plot_centers = (bins[:-1] + bins[1:])/2
    plot_values = group[yvar].mean()
    dd = pd.DataFrame(plot_centers, columns=[distval])

    dd[yvar] = plot_values.to_list()
    dd = dd.dropna()

    return dd


def get_radius(grp, distval='dist_center', yvar='nvisits'):

    df_dist = get_dist(grp)

    dda = rebin(df_dist)
    ddb = rebin(df_dist, yvar='cadence')

    print(df_dist)
    fig, ax = plt.subplots(nrows=2)
    tt = np.array(np.cumsum(dda[yvar]))
    # nt = np.cumsum(dd[yvar])[-1]
    ax[0].plot(dda[distval], tt/tt[-1], color='k', marker='o')
    ax[1].plot(ddb[distval], ddb['cadence'], color='k', marker='o')
    plt.show()
    print(test)


parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbList', type=str, default='list_db.csv',
                  help='dblist to process [%default]')
parser.add_option('--dbDir', type=str, default='../dd_pixels',
                  help='dbDir of the OS to process [%default]')

opts, args = parser.parse_args()

dbList = opts.dbList
dbDir = opts.dbDir

# load dbList
df_db = pd.read_csv(dbList, comment='#')

for i, row in df_db.iterrows():
    dbName = row['dbName']
    dbName_dir = '{}/{}'.format(dbDir, dbName)
    data = load_db(dbName_dir)
    print(data)
    idx = data['year'] >= 0
    data = data[idx]
    dd = data.groupby(['field', 'year']).apply(
        lambda x: get_radius(x), include_groups=False).reset_index()
