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

filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))


def ana_gaps(df):

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

    idx = df[colName].isin(llist)

    sel = df[idx]

    if len(sel) > 0:
        print(name, sel['mjd_diff'].mean(), sel['mjd_diff'].std())
    else:
        print(name, 0.0, 0.0)


def select(df, tmin, tmax):

    idx = df['mjd_diff'] >= tmin
    idx &= df['mjd_diff'] <= tmax

    sel = df[idx]

    res = sel['observationId'].to_list()

    return res


def ana_night(grp, mjdCol='mjd'):

    dd = grp.sort_values(by=['mjd'])

    fig, ax = plt.subplots()

    dd['mjd_diff'] = dd['mjd'].diff()
    dd['mjd_diff'] *= 24.*3600

    ana_gaps(dd)

    for b in 'ugrizy':
        idx = dd['filter'] == b
        sel = dd[idx]
        ax.plot(sel['mjd'], sel['mjd_diff'], color=filtercolors[b],
                marker='o', linestyle='None')

    figb, axb = plt.subplots()

    axb.hist(dd['mjd_diff'], histtype='step', bins=1000)

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

print(tt.dtype)

df = pd.DataFrame.from_records(tt)

print(df.columns)

dfb = df.groupby(['night']).apply(lambda x: ana_night(x))
