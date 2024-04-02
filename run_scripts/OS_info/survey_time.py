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
import os
from optparse import OptionParser


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


def process_survey_time(dbDir, configFile):

    df = process(dbDir, configFile)

    df.to_hdf(outName, key='sn')


def print_survey_time(df):

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
        '\\caption{Exposure times (in hours) for seasons 2-10. \
            $\Phi_{Moon}$ is the Moon phase.}\\label{tab:exptime_1}')
    print('\\begin{tabular}{l|c|c|c|c|c}')
    print('\hline')
    print('\hline')
    print(' & \multicolumn{4}{|c|}{nightly} & \\\\')
    print(
        'Strategy & \multicolumn{2}{|c|}{UDF} & \
            \multicolumn{2}{|c|}{DF} & survey \\\\')
    print(' & $\Phi_{Moon}\leq 20\\%$ & $\Phi_{Moon} > 20\\%$ \
          & $\Phi_{Moon}\leq 20\\%$ & $\Phi_{Moon} > 20\\% $& \\\\')

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


def estimate_survey_time(dbDir, configFile):

    outName = 'survey_time.hdf5'

    if not os.path.isfile(outName):

        df = process_survey_time(dbDir, configFile)

    df = pd.read_hdf(outName)

    print_survey_time(df)


def process_OS_depths(dbDir, configFile):

    conf = pd.read_csv(configFile)

    df = pd.DataFrame()

    for i, row in conf.iterrows():
        dfa = process_OS_depth(dbDir, row['dbName'])
        dfa['dbName'] = row['dbName']
        dfa['dbNamePlot'] = row['dbNamePlot']
        df = pd.concat((df, dfa))

    return df


def process_OS_depth(dbDir, dbName):

    full_path = '{}/{}.npy'.format(dbDir, dbName)

    data = np.load(full_path, allow_pickle=True)

    data = pd.DataFrame.from_records(season(data))

    idx = data['season'] == 1

    res = pd.DataFrame()

    for seas in data['season'].unique():
        idx = data['season'] == seas

        sela = data[idx]

        res_y = sela.groupby(['note', 'filter']).apply(
            lambda x: gime_m5_visits(x)).reset_index()
        res_y['season'] = seas

        res = pd.concat((res, res_y))

    # cumulate seasons 2-10
    idx = data['season'] >= 2
    selb = data[idx]
    res_c = selb.groupby(['note', 'filter']).apply(
        lambda x: gime_m5_visits(x)).reset_index()
    res_c['season'] = 11

    res = pd.concat((res, res_c))

    return res


def estimate_depth(dbDir, configFile):

    # process the data
    df = process_OS_depths(dbDir, configFile)

    print_latex_depth(df)


def gime_m5_visits(grp):

    m5_coadd = 1.25*np.log10(np.sum(10**(0.8*grp['fiveSigmaDepth'])))
    nvisits = len(grp)

    res = pd.DataFrame({'m5': [m5_coadd], 'nvisits': [nvisits]})

    return res


def print_latex_depth(df):

    dbNames = df['dbName'].unique()

    for io, dbName in enumerate(dbNames):
        idx = df['dbName'] == dbName
        sel = df[idx]
        dbNameb = sel['dbNamePlot'].unique()[0]
        print_latex_depth_os(sel, dbName, io, dbNameb)


def print_latex_depth_os(df, dbName, io, dbNameb):

    tta = ['DD:COSMOS', 'DD:ELAISS1', 'DD:XMM_LSS',
           'DD:ECDFS', 'DD:EDFS_a', 'DD:EDFS_b']
    ttb = ['\cosmos', '\elais', '\\xmm', '\cdfs', '\\adfa', '\\adfb']
    trans_ddf = dict(zip(tta, ttb))

    fields = df['note'].unique()

    bands = list('ugrizy')
    dbNameb = dbNameb.split('_')
    dbNameb = '\_'.join(dbNameb)
    caption = '{} strategy: coadded \\fivesig~depth and total number of visits $N_v$ per band.'.format(
        dbNameb)

    caption = '{'+caption+'}'
    label = 'tab:total_depth_{}'.format(io)
    label = '{'+label+'}'
    r = get_beg_table(caption=caption, label=label)
    rr = ' & '
    pp = 'Field & '
    seas_max = 10
    # for seas in df['season'].unique():
    for seas in [1]:
        pp += 'season & $m_5$ &$N_v$'
        bb = '/'.join(bands)
        rr += ' & {} & {}'.format(bb, bb)
        rr += ' \\\\'
        pp += ' \\\\'
        """
        if seas < seas_max:
            rr += ' & '
            pp += ' & '
        else:
            rr += ' \\\\'
            pp += ' \\\\'
        """
        r += [pp]
        r += [rr]
        r += [' & & & \\\\']
        r += ['\hline']
    for fi in fields:
        idx = df['note'] == fi
        selb = df[idx]
        ll = print_latex_depth_field(selb)
        for io, vv in enumerate(ll):
            if io != 5:
                tt = ' & {}'.format(vv)
            else:
                tt = '{} & {}'.format(trans_ddf[fi], vv)
            r += [tt]
        r += ['\hline']
    r += get_end_table()

    for vv in r:
        print(vv)


def print_latex_depth_field(data, bands='ugrizy'):

    r = []
    seasons = range(1, 12, 1)
    seas_tt = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '2-10']
    trans_seas = dict(zip(seasons, seas_tt))

    for seas in data['season'].unique():
        idx = data['season'] == seas
        sel = data[idx]
        m5 = []
        nv = []
        for b in list(bands):
            idxb = sel['filter'] == b
            selb = sel[idxb]
            mm5 = selb['m5'].values[0]
            nvv = selb['nvisits'].values[0]
            m5.append('{}'.format(np.round(mm5, 1)))
            nv.append('{}'.format(int(nvv)))
        m5_tot = '/'.join(m5)
        nv_tot = '/'.join(nv)

        rr = '{} & {} & {}'.format(trans_seas[seas], m5_tot, nv_tot)
        rr += '\\\\'
        r.append(rr)
        """
        if seas != 10:
            rr += ' & '
        else:
            rr += ' \\\\'
        """
    return r


def get_beg_table(caption='{test}', label='{tab:test}'):

    r = ['\\begin{table*}[!htbp]']
    r += ['\\tiny']
    r += ['\\begin{center}']
    r += ['\caption{} \label{}'.format(caption, label)]
    # r += ['\\begin{tabular}{l|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c}']
    r += ['\\begin{tabular}{l|c|c|c}']
    r += ['\hline']
    r += ['\hline']
    r += [' & & & \\\\']
    return r


def get_end_table():

    r = ['\hline']
    r += ['\end{tabular}']
    r += ['\end{center}']
    r += ['\end{table*}']

    return r


parser = OptionParser()

parser.add_option("--dbDir", type=str,
                  default='../DB_Files',
                  help="data dir[%default]")
parser.add_option("--configFile", type=str,
                  default='config_ana_paper_plot.csv', help="configuration file [%default]")
parser.add_option("--what", type=str,
                  default='survey_time,depth', help="what to estimate [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
configFile = opts.configFile
whats = opts.what.split(',')


if 'survey_time' in whats:
    estimate_survey_time(dbDir, configFile)

if 'depth' in whats:
    estimate_depth(dbDir, configFile)


"""
idx = df['season'] == 1
sela = df[idx]

print(sela)
"""
