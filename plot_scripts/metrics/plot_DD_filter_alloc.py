#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:12:40 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_plotter_metrics.plot4metric import plot_filter_alloc
from optparse import OptionParser
import pandas as pd
from sn_plotter_metrics import plt


def load_data(dirFile, dbList):
    """
    Function to load data

    Parameters
    ----------
    dirFile : str
        dir files.
    dbList : list(str)
        db list.

    Returns
    -------
    data : pandas df
        output data.

    """

    data = pd.DataFrame()
    for dbName in dbList:
        fName = '{}/{}/Summary_DD_pointings.hdf5'.format(dirFile, dbName)
        print(fName)
        df_ = pd.read_hdf(fName)
        # print(df_.columns)
        df_['dbName'] = dbName

        data = pd.concat((data, df_))

    return data


def complete_pointing(df, dfgroup):
    """
    function to merge two df, make some cleaning, ...

    Parameters
    ---------------
    df: pandas df
      first pandas df
    dfgroup: pandas df
      second pandas df

    Returns
    ------------
    modified merged df

    """
    df = df.merge(dfgroup[['dbName', 'dbName_plot', 'marker', 'color', 'ls']],
                  left_on=['dbName'], right_on=['dbName'])

    # df['family'] = df['group']

    # strip db Name
    # df['family'] = df['family'].str.split('_v2.99_10yrs', expand=True)[0]

    # uniformity of DD names

    torep = dict(zip(['ECDFS', 'EDFS, a', 'EDFS, b', 'EDFS_a', 'EDFS_b', 'XMM_LSS'], [
        'CDFS', 'EDFSa', 'EDFSb', 'EDFSa', 'EDFSb', 'XMM-LSS']))

    """
    for key, vals in torep.items():
        df['field'] = df['field'].str.replace(
            key, vals)
    df['field'] = df['field'].str.split(':', expand=True)[1]
    """

    return df


def flat_this(grp, cols=['filter_alloc', 'filter_frac']):
    """
    Function to flatten some df columns

    Parameters
    ----------
    grp : pandas df
        data to process
    cols : list(str), optional
        list of cols to flatten. The default is ['filter_alloc', 'filter_frac'].

    Returns
    -------
    pandas df
        data with flattened cols

    """

    dictout = {}

    for vv in cols:
        dictout[vv] = sum(grp[vv].to_list(), [])

    return pd.DataFrame.from_dict(dictout)


parser = OptionParser(
    description='OS filter sequence from pointings')

parser.add_option("--dirFile", type="str",
                  default='../summary_DD_pointings',
                  help="file directory [%default]")
parser.add_option("--pointingFile", type="str",
                  default='Summary_DD_pointings.hdf5',
                  help="pointing file name [%default]")
parser.add_option("--config", type="str", default='DD_fbs_2.99_plot.csv',
                  help="pointing file name [%default]")
parser.add_option("--addMetric", type=int, default=0,
                  help="to add metric correlation plots [%default]")
parser.add_option("--dbName_night", type=str, default='baseline_v5.0.0_10yrs',
                  help="dbName for night plot stat [%default]")
parser.add_option("--fieldName_night", type=str, default='COSMOS',
                  help="field for night plot stat [%default]")
parser.add_option("--plots", type=str,
                  default='filter_alloc',
                  help="plots to draw [%default]")

opts, args = parser.parse_args()
# Load parameters
dirFile = opts.dirFile
# dbList = opts.dbList
pointingFile = opts.pointingFile
config = opts.config
dbName_night = opts.dbName_night
fieldName_night = opts.fieldName_night
plots = opts.plots.split(',')

df_conf = pd.read_csv(config, comment='#')  # load list of db+plot infos
# df = pd.read_hdf(pointingFile)  # load pointing data
df = load_data(dirFile, df_conf['dbName'].to_list())

df = complete_pointing(df, df_conf)  # merge pointing data+plot data


if 'filter_alloc' in plots:
    flat = df.groupby(['dbName', 'dbName_plot', 'field', 'season']).apply(
        lambda x: flat_this(x, cols=['filter_alloc', 'filter_frac']),
        include_groups=False).reset_index()

    flat = flat.groupby(['dbName', 'dbName_plot', 'field', 'filter_alloc', 'season'])[
        'filter_frac'].median().reset_index()

    idx = df_conf['dbName'] == dbName_night

    family = df_conf[idx]['dbName_plot'].to_list()[0]

    seasons = flat['season'].unique()
    for seas in seasons:
        plot_filter_alloc(flat, family, fieldName_night, season=seas)

if len(plots) > 0:
    plt.show()
