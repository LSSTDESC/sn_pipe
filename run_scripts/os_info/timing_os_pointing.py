#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:53:23 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from optparse import OptionParser
from sn_tools.sn_obs import get_fields
filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))


def ana_gaps(df):
    """
    Function to analyze time gaps

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    None.

    """

    # filter changes

    r = filter_swap(df)
    info(df, r, name='filter swap')

    # less than 20 sec exposures

    resa = select(df, 10., 25.)
    r += resa
    info(df, resa, name='exposure 15 sec')

    # exposure 30 sec

    resb = select(df, 30, 45)

    r += resb
    info(df, resb, name='exposure 30s')

    idx = df['observationId'].isin(r)

    sel_fi = df[~idx]

    if len(sel_fi) > 0:
        print('alors?', sel_fi['mjd_diff'])


def filter_swap(df):
    """
    Function to estimete filter swap timing

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    r : list(int)
        List of obsId corresponding to filter swaps.

    """

    r = []
    filt_prev = 'u'

    for i, row in df.iterrows():
        filt = row['filter']
        delta_mjd = row['mjd_diff']
        if filt != filt_prev and delta_mjd < 300.:
            r.append(row['observationId'])
        filt_prev = filt

    return r


def info(df, llist, colName='observationId', name='what'):
    """
    Function to estimate mean and std

    Parameters
    ----------
    df : pandas df
        Data to process.
    llist : list(int)
        list of obs id.
    colName : str, optional
        column for selection. The default is 'observationId'.
    name : str, optional
        config for estimation. The default is 'what'.

    Returns
    -------
    None.

    """

    idx = df[colName].isin(llist)

    sel = df[idx]

    if len(sel) > 0:
        print(name, sel['mjd_diff'].mean(), sel['mjd_diff'].std())
    else:
        print(name, 0.0, 0.0)


def select(df, tmin, tmax, sel_colName='mjd_diff', out_colName='observationId'):
    """
    data selection

    Parameters
    ----------
    df : pandas df
        Data to process.
    tmin : float
        time min.
    tmax : float
        time max.
    sel_colName : str, optional
        column name for selection. The default is 'mjd_diff'.
    out_colName : str, optional
        col name for output list. The default is 'observationId'.

    Returns
    -------
    res : list(int)
        list of out_colName selected.

    """

    idx = df[sel_colName] >= tmin
    idx &= df[sel_colName] <= tmax

    sel = df[idx]

    res = sel[out_colName].to_list()

    return res


def ana_night(grp, mjdCol='mjd'):
    """
    Function to analyze an observing night

    Parameters
    ----------
    grp : pandas df
        data to process.
    mjdCol : str, optional
        MJD column name. The default is 'mjd'.

    Returns
    -------
    None.

    """

    dd = grp.sort_values(by=['mjd'])

    dd['mjd_diff'] = dd['mjd'].diff()
    dd['mjd_diff'] *= 24.*3600

    ana_gaps(dd)

    plot(dd)


def plot(dd):

    fig, ax = plt.subplots()

    markers = dict(zip(['WFD', 'DDF'], ['o', 's']))
    for b in 'ugrizy':
        idx = dd['filter'] == b
        sel = dd[idx]
        for ftype in ['WFD', 'DDF']:
            idxb = sel['fieldType'] == ftype
            selb = sel[idxb]
            print(ftype, selb[['fieldType', 'scheduler_note']])
            ax.plot(selb['mjd'], selb['mjd_diff'], color=filtercolors[b],
                    marker=markers[ftype], linestyle='None', mfc='None')

    """
    figb, axb = plt.subplots()

    axb.hist(dd['mjd_diff'], histtype='step', bins=1000)
    """
    plt.show()


parser = OptionParser(description='survey timing analysis')
parser.add_option("--dbDir", type="str",
                  default='../DB_Files',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='baseline_v5.0.0_10yrs',
                  help="file directory [%default]")

opts, args = parser.parse_args()
dbDir = opts.dbDir
dbName = opts.dbName

fName = '{}/{}.npy'.format(dbDir, dbName)

tt = np.load(fName)

# get DDFs
ddf = get_fields(tt, 'input/simulation/lookup_ddf.csv')

print(ddf)

ddf_obsid = ddf['observationId'].tolist()
print(ddf_obsid)

print(tt.dtype)

df = pd.DataFrame.from_records(tt)

df['fieldType'] = 'WFD'

idx = df['observationId'].isin(ddf_obsid)

df.loc[idx, 'fieldType'] = 'DDF'

print(df.columns)

df = df.sort_values(by=['night'])
dfb = df.groupby(['night']).apply(lambda x: ana_night(x))
