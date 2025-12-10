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
from scipy.interpolate import interp1d


def load_db(dbDir):
    """
    Function to load db data

    Parameters
    ----------
    dbDir : str
        OS location dir.

    Returns
    -------
    df : pandas df
        loaded data.

    """

    search_path = '{}/*.hdf5'.format(dbDir)
    print('path', search_path)
    fis = glob.glob(search_path)

    df = pd.DataFrame()

    for fi in fis:
        tt = pd.read_hdf(fi)
        df = pd.concat((df, tt))

    return df


def rebin(df_dist, distval='dist_center', yvar='cadence', xmin=0.0, xmax=3.5, nbins=15):
    """
    Function used to rebin data

    Parameters
    ----------
    df_dist : pandas df
        Data to rebin.
    distval : str, optional
        x-axis value. The default is 'dist_center'.
    yvar : str, optional
        y-axis value. The default is 'nvisits'.
    nbins: int, optional
        number of bins for rebining. The defauls is 15.

    Returns
    -------
    dd : pandas df
        Rebinned data.

    """

    # rebin to have a "better" plot
    # xmin, xmax = df_dist[distval].min(), df_dist[distval].max()
    bins = np.linspace(xmin, xmax, nbins)
    # bins = np.arange(0.1, 2.22, 0.22)
    group = df_dist.groupby(pd.cut(df_dist[distval], bins), observed=False)
    plot_centers = (bins[:-1] + bins[1:])/2
    plot_values = group[yvar].mean()
    plot_std = group[yvar].std().to_list()

    dd = pd.DataFrame(plot_centers, columns=[distval])

    dd['{}_mean'.format(yvar)] = plot_values.to_list()
    dd['{}_std'.format(yvar)] = plot_std

    if yvar == 'nvisits':
        plot_values = group[yvar].sum()
        ntot = np.sum(plot_values)
        dd['{}_sum'.format(yvar)] = plot_values.to_list()
        dd['{}_tot'.format(yvar)] = ntot
    dd = dd.dropna()

    return dd


def get_radius(grp, distval='dist_center', yvar=['nvisits', 'cadence'], nbins=15):
    """
    Function to estimate nvisits and cadence vs radius

    Parameters
    ----------
    grp : pandas df
        Data to process.
    distval : str, optional
        distance to use. The default is 'dist_center'.
    yvar : str, optional
        columns to estimate. The default is ['nvisits','cadence'].
    nbins: int, optional
        number of bins for rebining. The defauls is 15.
    Returns
    -------
    ddc : pandas df
        output data.

    """

    # print('processing', grp.name, grp)

    df_dist = get_dist(grp)

    idx = df_dist['cadence'] > 0
    df_dist = df_dist[idx]
    """
    fig, ax = plt.subplots()
    ax.plot(df_dist[distval], df_dist['cadence'])
    plt.show()
    """
    dd = {}
    for i, val in enumerate(yvar):
        dd[i] = rebin(df_dist, yvar=val, nbins=nbins)

    ddc = dd[0].merge(dd[1], left_on=[distval], right_on=[
        distval], suffixes=['', ''])
    return ddc

    """
    idx = ddb['cadence'] > 0.
    ddb = ddb[idx]

    idb = ddb[distval] <= 0.7  # selection at 0.7 deg
    mean_cad = ddb[idb]['cadence'].mean()

    print(df_dist)
    fig, ax = plt.subplots(nrows=2)
    tt = np.array(np.cumsum(dda[yvar]))
    # nt = np.cumsum(dd[yvar])[-1]
    ax[0].plot(dda[distval], tt/tt[-1], color='k', marker='o')
    ax[1].plot(ddb[distval], ddb['cadence']-mean_cad, color='k', marker='o')

    vv = tt/tt[-1]
    interp_nvisits = interp1d(
        dda[distval], vv, bounds_error=False, fill_value=0)
    interp_cadence = interp1d(
        ddb[distval], ddb['cadence']-mean_cad, bounds_error=False, fill_value=0)

    ddist_max = np.max(df_dist['dist_center'])
    ddist = np.arange(0., ddist_max, 0.01)

    fig, ax = plt.subplots()
    ax.plot(interp_cadence(ddist), interp_nvisits(ddist),  'ko')

    plt.show()

    """


parser = OptionParser(description='Script to plot pixel level OS infos')

parser.add_option('--dbList', type=str, default='list_db.csv',
                  help='dblist to process [%default]')
parser.add_option('--dbDir', type=str, default='../dd_pixels',
                  help='dbDir of the OS to process [%default]')
parser.add_option('--outName', type=str, default='data_radius.hdf5',
                  help='output file name [%default]')
parser.add_option('--nbins', type=int, default=15,
                  help='nbins for rebinning [%default]')

opts, args = parser.parse_args()

dbList = opts.dbList
dbDir = opts.dbDir
outName = opts.outName
nbins = opts.nbins

# load dbList
df_db = pd.read_csv(dbList, comment='#')

df_tot = pd.DataFrame()

for i, row in df_db.iterrows():
    dbName = row['dbName']
    dbName_dir = '{}/{}'.format(dbDir, dbName)
    data = load_db(dbName_dir)
    idx = data['year'] >= 0
    idx &= data['year'] <= 10
    data = data[idx]
    dd = data.groupby(['field', 'year']).apply(
        lambda x: get_radius(x, nbins=nbins), include_groups=False).reset_index()
    dd['dbName'] = dbName
    df_tot = pd.concat((df_tot, dd))

# print(df_tot)
df_tot.to_hdf(outName, key='radius')
