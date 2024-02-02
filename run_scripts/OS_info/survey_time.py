#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:03:02 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
from sn_tools.sn_obs import season
import pandas as pd
import operator


def nvisits(grp, varsel, valsel, op):

    idx = op(grp[varsel], valsel)

    sel = grp[idx]

    nvisits = len(sel)
    expTime = sel['visitExposureTime'].sum()

    return pd.DataFrame([[expTime, len(sel)]], columns=['expTime', 'nvisits'])


def nightly_visits(data):

    # get the nightly number of visits
    nv_moon = data.groupby(['note', 'night']).apply(
        lambda x: nvisits(x, 'moonPhase', 20, operator.gt))
    nv_nomoon = data.groupby(['note', 'night']).apply(
        lambda x: nvisits(x, 'moonPhase', 20, operator.le))

    dfa = calc_exptime(nv_moon)
    dfb = calc_exptime(nv_nomoon)

    dfa['moon'] = 1
    dfb['moon'] = 0

    df = pd.concat((dfa, dfb))

    return df


def calc_exptime(data):

    idx = data['expTime'] > 0.
    sel_data = data[idx]

    med_exptime = sel_data['expTime'].median()
    sum_exptime = sel_data['expTime'].sum()

    return pd.DataFrame([[med_exptime, sum_exptime]], columns=['expTime_nightly', 'expTime_sum'])


def get_infos(data, season, field='all'):

    dfb = pd.DataFrame()
    for seas in season:
        idx = data['season'] == seas
        if field != 'all':
            idx &= data['note'] == 'DD:{}'.format(field)
        data_y = data[idx]

        df_y = nightly_visits(data_y)

        df_y['season'] = seas
        df_y['field'] = field

        dfb = pd.concat((dfb, df_y))

    return dfb


def process_OS(dbDir, dbName):

    full_path = '{}/{}.npy'.format(dbDir, dbName)

    data = np.load(full_path, allow_pickle=True)

    data = pd.DataFrame.from_records(season(data))

    # season 1 stat
    df_y1 = get_infos(data, [1])

    # season 2 fields
    df_y2 = pd.DataFrame()
    for vv in ['COSMOS', 'ELAISS1', 'EDFS_a']:
        df_y = get_infos(data, [2], vv)
        df_y2 = pd.concat((df_y, df_y2))

    # print(df_y2)
    # all fields, all seasons

    df_tot = pd.concat((df_y1, df_y2))
    # df_all = get_infos(data, range(2, 11))

    idx = data['season'] > 1
    # print(df_all['expTime_sum'].sum(), data[idx]['visitExposureTime'].sum())

    expTime = data[idx]['visitExposureTime'].sum()

    df_all = pd.DataFrame([expTime], columns=['expTime_sum'])
    df_all['season'] = 10
    df_all['field'] = 'all'
    df_all['moon'] = -1
    df_all['expTime_nightly'] = -1

    df_tot = pd.concat((df_tot, df_all))

    return df_tot


def process(dbDir, configFile):

    conf = pd.read_csv(configFile)

    df = pd.DataFrame()

    for i, row in conf.iterrows():
        dfa = process_OS(dbDir, row['dbName'])
        dfa['dbName'] = row['dbName']
        df = pd.concat((df, dfa))
        # break

    return df


def get_vals(sela, field):

    idxb = sela['field'] == field
    selb = sela[idxb]

    idxc = selb['moon'] == 0
    ra = selb[idxc]['expTime_nightly'].values[0]
    rb = selb[~idxc]['expTime_nightly'].values[0]

    ra /= 3600
    rb /= 3600.
    return [np.round(ra, 1), np.round(rb, 1)]


outName = 'survey_time.hdf5'
"""
dbDir = '../DB_Files'

configFile = 'config_ana_paper_plot.csv'

df = process(dbDir, configFile)

df.to_hdf(outName, key='sn')

"""

df = pd.read_hdf(outName)

"""
idx = df['season'] == 1
sela = df[idx]

print(sela)
"""

# udf
idx = df['season'] == 2
sel_y2 = df[idx]

idd = df['season'] == 10
sel_all = df[idd]


rtot = []
for dbName in sel_y2['dbName'].unique():
    idxa = sel_y2['dbName'] == dbName
    sela = sel_y2[idxa]
    r = [dbName]
    for vv in ['COSMOS', 'ELAISS1']:
        rr = get_vals(sela, vv)
        r += rr
    rtot.append(r)
    idf = sel_all['dbName'] == dbName
    r_all = sel_all[idf]['expTime_sum'].mean()
    r_all = np.round(r_all/3600., 1)
    r.append(r_all)
    rtot.append(r)
    # break
print(rtot)

print('\\begin{table}[!htbp]')
print('\\begin{center}')
print(
    '\\caption{Exposure times (in hours) for seasons 2-10. $\Phi_{Moon}$ is the Moon phase.}\\label{tab:exptime_1}')
print('\\begin{tabular}{l|c|c|c|c|c}')
print('\hline')
print('\hline')
print(' & \multicolumn{4}{|c|}{nightly} & \\\\')
print(
    'Strategy & \multicolumn{2}{|c|}{UDF} & \multicolumn{2}{|c|}{DF} & survey \\\\')
print(' & $\Phi_{Moon}\leq 20\\%$ & $\Phi_{Moon} > 20\\%$ & $\Phi_{Moon}\leq 20\\%$ & $\Phi_{Moon} > 20\\% $& \\\\')

print('\hline')

for vv in rtot:
    dbName = '_'.join(vv[0].split('_')[:-1])
    dbName = dbName.replace('_', '\_')
    print('{} & {} & {} & {} & {} & {} \\\\'.format(
        dbName, vv[1], vv[2], vv[3], vv[4], vv[5]))

print('\hline')
print('\hline')
print('\end{tabular}')
print('\end{center}')
print('\end{table}')
