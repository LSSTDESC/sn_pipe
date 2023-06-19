#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:34:46 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
import pandas as pd
from optparse import OptionParser
from brokenaxes import brokenaxes
from sn_plotter_analysis import plt
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
from sn_tools.sn_io import checkDir
from sn_tools.sn_obs import season


def coadd_night(grp, colsum=['numExposures'],
                colmean=['season', 'observationStartMJD', 'moonPhase'],
                combi_filt=['ugriz', 'grizy']):
    """
    Function to coadd obs per night

    Parameters
    ----------
    grp : pandas df
        data to ,process.
    colsum : list(str), optional
        List of col for which to perform a sum. The default is ['numExposures'].
    colmean : list(str), optional
        List of cols for wwhich to estimate the mean.
        The default is ['season', 'observationStartMJD'].

    Returns
    -------
    pandas df
        output res.

    """

    dictout = {}
    for vv in colsum:
        dictout[vv] = [grp[vv].sum()]

    for vv in colmean:
        dictout[vv] = [grp[vv].mean()]

    filters = list(grp['filter'].unique())
    filters.sort()

    for cc in combi_filt:
        vv = list(cc)
        vv.sort()
        if vv == filters:
            dictout['filter_alloc'] = ['/'.join(cc)]
            nf = []
            for ff in list(cc):
                idx = grp['filter'] == ff
                nf.append(grp[idx]['numExposures'].sum())

            nf = map(str, nf)
            dictout['visits_band'] = ['/'.join(nf)]

    return pd.DataFrame.from_dict(dictout)


def coadd_season(grp):
    """
    print('allo', np.unique(grp[['filter_alloc', 'visits_band']]), len(grp))
    filter_alloc = grp['filter_alloc'].to_list()
    visits_band = grp['visits_band'].to_list()
    """

    filter_alloc = []
    visits_band = []

    falloc_all = grp['filter_alloc'].unique()
    for ff in np.sort(falloc_all):
        filter_alloc.append(ff)
        idx = grp['filter_alloc'] == ff
        visits_band.append(grp[idx]['visits_band'].unique()[0])

    visits_band = map(str, visits_band)
    dictout = {}

    dictout['filter_alloc'] = ['_or_'.join(filter_alloc)]
    dictout['visits_band'] = ['_or_'.join(visits_band)]

    vv = 'observationStartMJD'
    grp = grp.sort_values(by=[vv])

    cadence = np.mean(grp[vv].diff())

    dictout['cadence'] = [cadence]
    Tmin = grp[vv].min()
    Tmax = grp[vv].max()

    dictout['Tmin'] = [np.round(Tmin, 3)]
    dictout['Tmax'] = [np.round(Tmax, 3)]

    return pd.DataFrame.from_dict(dictout)


def coadd_field(grp):

    fields = list(grp['note'].unique())
    dictout = {}
    dictout['note'] = [','.join(fields)]

    return pd.DataFrame.from_dict(dictout)


def coadd_final(grp):

    seas_min = grp['season'].min()
    seas_max = grp['season'].max()
    Tmin = grp['Tmin'].min()
    Tmax = grp['Tmax'].max()

    seas_min -= 1

    dictout = {}
    dictout['seas_min'] = [seas_min]
    dictout['seas_max'] = [seas_max]
    dictout['Tmin'] = [Tmin]
    dictout['Tmax'] = [Tmax]

    return pd.DataFrame.from_dict(dictout)


def translate(grp):

    grp = grp.sort_values(by=['season'])
    seasons = grp['season'].unique()

    val = 'observationStartMJD'
    Tmax = grp[grp['season'] == 1][val].max()
    night_max = grp[grp['night'] == 1][val].max()
    dfi = pd.DataFrame()
    for seas in seasons:
        idx = grp['season'] == seas
        sel = pd.DataFrame(grp[idx])
        Tmin = sel[val].min()
        night_min = sel['night'].min()
        deltaT = Tmin-Tmax
        deltaN = night_min-night_max
        if seas == 1:
            deltaT = 0.
            deltaN = 0
        sel.loc[:, val] -= deltaT
        sel.loc[:, 'night'] -= deltaN

        Tmax = sel[val].max()
        night_max = sel['night'].max()
        TTmin = Tmin-deltaT
        NNmin = night_min-deltaN
        sel['MJD_season'] = (sel[val]-TTmin)/(Tmax-TTmin)+(seas-1)
        sel['night_season'] = (sel[val]-NNmin)/(night_max-NNmin)+(seas-1)
        dfi = pd.concat((dfi, sel))

    return dfi


def doInt(df, cols):

    for cc in cols:
        df[cc] = df[cc].astype(int)

    return df


def nmax(list_visit):

    ro = []
    for ll in list_visit:
        vv = ll.split('/')
        vv = list(map(int, vv))
        ro.append(np.sum(vv))

    return np.max(ro)


def gime_combi(filter_alloc, nvisits_band):

    dictout = {}
    for ia, ff in enumerate(filter_alloc):
        fa = ff.split('/')
        nv = nvisits_band[ia].split('/')
        nv = list(map(int, nv))

        for i, vv in enumerate(fa):
            if vv not in dictout.keys():
                dictout[vv] = []
            dictout[vv].append(nv[i])

    rb = []
    for key, vals in dictout.items():
        rb.append((key, np.max(vals)))

    res = np.rec.fromrecords(rb, names=['band', 'visits_band'])

    rf = []
    rb = []
    combi1 = []
    combi2 = []
    for b in 'ugrizy':
        idx = res['band'] == b
        rf.append(b)
        nv = int(np.mean(res[idx]['visits_band']))
        rb.append(nv)
        if b == 'u' or b == 'g' or b == 'r':
            combi1.append('{}{}'.format(nv, b))
        else:
            combi2.append('{}{}'.format(nv, b))

    sf = '/'.join(rf)
    nvis = '/'.join(map(str, rb))
    combi1 = '/'.join(combi1)
    combi2 = '/'.join(combi2)

    return sf, nvis, combi1, combi2


def plot_resu(df_all, df_coadd, dbName, outDir):

    configs = df_coadd['note'].unique()
    vala = 'observationStartMJD'
    valb = 'numExposures'
    valc = 'MJD_season'
    fig, ax = plt.subplots(nrows=len(configs), figsize=(16, 9))
    fig.suptitle(dbName, fontweight='bold')
    fig.subplots_adjust(wspace=0., hspace=0.)
    configs = np.sort(configs)
    for i, conf in enumerate(configs):

        idx = df_coadd['note'] == conf
        sel = df_coadd[idx]

        nn = conf.split(',')[0]

        idxb = df_all['note'] == nn
        sel_all = df_all[idxb]
        sel_all = translate(sel_all)
        print('hh', sel_all.columns)
        idm = sel_all['moonPhase'] <= 20.
        ax[i].plot(sel_all[idm][valc], sel_all[idm]
                   [valb], 'ko', mfc='None', ms=4)
        idm = sel_all['moonPhase'] > 20.
        ax[i].plot(sel_all[idm][valc], sel_all[idm]
                   [valb], 'k*', mfc='None', ms=4)

        ymin, ymax = ax[i].get_ylim()
        rymax = []
        for ib, row in sel.iterrows():
            seas_min = row['seas_min']
            seas_max = row['seas_max']
            cad = row['cadence']
            filter_alloc = row['filter_alloc'].split('_or_')
            visits_band = row['visits_band'].split('_or_')
            cadence = row['cadence']
            print(filter_alloc)
            print(visits_band)
            print(cadence)
            ymax = nmax(visits_band)
            rymax.append(ymax)
            ax[i].plot([seas_min]*2, [ymin, ymax],
                       linestyle='dashed', color='k')
            ax[i].plot([seas_max]*2, [ymin, ymax],
                       linestyle='dashed', color='k')
            faloc, nvis, combi1, combi2 = gime_combi(filter_alloc, visits_band)
            seas_mean = 0.5*(seas_min+seas_max)
            yyy = 0.5*(ymax-ymin)+ymin
            k = 0.08

            if seas_min == 0:
                k = 0.01

            ax[i].text(k*seas_mean, 0.65,
                       combi1, color='b',
                       fontsize=12, transform=ax[i].transAxes)

            ax[i].text(k*seas_mean, 0.55,
                       combi2, color='b',
                       fontsize=12, transform=ax[i].transAxes)

            ax[i].text(k*seas_mean, 0.45,
                       'cadence = {}'.format(cadence), color='b',
                       fontsize=12, transform=ax[i].transAxes)
        ll = range(1, 11, 1)
        ax[i].set_xticks(list(ll))
        if i == len(configs)-1:
            for it in ll:
                ax[i].text(0.1*(it-0.5), -0.15, '{}'.format(it),
                           transform=ax[i].transAxes)
            #ax[i].set_xlabel('Season', labelpad=15)
            ax[i].text(0.5, -0.30, 'Season',
                       transform=ax[i].transAxes, ha='center')

        ax[i].set_ylabel('N$_{visits}$')
        ax[i].set_xlim([0, 10])
        ax[i].text(0.95, 0.8, conf, color='r',
                   fontsize=15, transform=ax[i].transAxes, ha='right')
        ax[i].grid()
        ax[i].tick_params(axis='x', colors='white')

    outName = '{}/cadence_{}.png'.format(outDir, dbName)
    plt.savefig(outName)
    plt.close(fig)


def plot_cadence(dbDir, dbName, df_config_scen, outDir):

    idx = df_config_scen['scen'] == dbName

    dname = df_config_scen[idx][['field', 'fieldType']]
    dname = dname.rename(columns={'field': 'note'})

    data = np.load('{}/{}.npy'.format(dbDir, dbName))
    df = pd.DataFrame.from_records(data)

    df = df.merge(dname, left_on=['note'], right_on=['note'])

    dfb = df.groupby(['note', 'fieldType', 'night']).apply(
        lambda x: coadd_night(x)).reset_index()

    dfb = doInt(dfb, ['season'])
    print(dfb)

    dfc = dfb.groupby(['note', 'fieldType', 'season']).apply(
        lambda x: coadd_season(x)).reset_index()

    dfc = doInt(dfc, ['season', 'cadence'])
    print(dfc)

    dfd = dfc.groupby(['fieldType', 'season', 'filter_alloc', 'visits_band',
                       'cadence', 'Tmin', 'Tmax']).apply(lambda x: coadd_field(x)).reset_index()

    dfd = doInt(dfd, ['season', 'cadence'])

    print(dfd[['fieldType', 'season', 'filter_alloc', 'visits_band',
              'cadence', 'note', 'Tmin', 'Tmax']])

    dfe = dfd.groupby(['fieldType', 'note', 'filter_alloc', 'visits_band',
                       'cadence']).apply(lambda x: coadd_final(x)).reset_index()
    dfe = doInt(dfe, ['cadence'])

    print(dfe[['fieldType', 'seas_min', 'seas_max', 'filter_alloc', 'visits_band',
              'cadence', 'note']])

    plot_resu(dfb, dfe, dbName.split('.npy')[0], outDir)


def plot_budget(dbDir, df_config_scen, Nvisits_LSST):

    dbNames = list(df_config_scen['scen'].unique())

    vala = 'observationStartMJD'
    valb = 'numExposures'
    valc = 'MJD_season'
    norm = Nvisits_LSST

    fig, ax = plt.subplots()
    for dbName in dbNames:
        data = np.load('{}/{}.npy'.format(dbDir, dbName))
        data = pd.DataFrame.from_records(data)

        # sum numexposures by night
        dt = data.groupby(['night']).apply(
            lambda x: coadd_night(x)).reset_index()
        print(dt['season'].unique())
        dt = dt.sort_values(by=['night'])
        dt[valb] /= norm
        # re-estimate seasons
        # dt = dt.drop(columns=['season'])
        #dt = season(dt.to_records(index=False), mjdCol=vala)
        selt = pd.DataFrame.from_records(dt)

        selt = translate(selt)
        selt = selt.sort_values(by=[valc])
        ax.plot(selt[valc], np.cumsum(selt[valb]), label=dbName)

    ax.legend()


parser = OptionParser(description='Script to analyze Observing Strategy')

parser.add_option('--dbDir', type=str, default='../DB_Files',
                  help='OS location dir [%default]')
parser.add_option('--configScenario', type='str',
                  default='input/DESC_cohesive_strategy/config_scenarios.csv',
                  help='config file to use[%default]')
parser.add_option('--outDir', type=str, default='Plot_OS',
                  help='Where to store the plots [%default]')
parser.add_option('--Nvisits_LSST', type=float, default=2.1e6,
                  help='Total number of LSST visits[%default]')


opts, args = parser.parse_args()

dbDir = opts.dbDir
configScenario = opts.configScenario
outDir = opts.outDir
Nvisits_LSST = opts.Nvisits_LSST

checkDir(outDir)
df_config_scen = pd.read_csv(configScenario, comment='#')

# plot cadence
"""
dbNames = df_config_scen['scen'].unique()
for dbName in dbNames:
    plot_cadence(dbDir, dbName, df_config_scen, outDir)

"""
# plot budget vs season
plot_budget(dbDir, df_config_scen, Nvisits_LSST)
plt.show()
print(test)
fields = dfb['note'].unique()
print(dfb.columns, fields)

ff = ['DD:COSMOS', 'DD:XMM_LSS',
      'DD:ECDFS', 'DD:EDFS_a',
      'DD:EDFS_b', 'DD:ELAISS1']

"""
mk = dict(zip(ff, ['o']*2+['*']*4))
colors = dict(zip(ff, ['k']*2+['r']*4))
fig, ax = plt.subplots(figsize=(14, 9))

val = 'observationStartMJD'
ymax = 1.05*dfb['numExposures'].max()
xmin = dfb[val].min()
# fig, (ax, ax2) = plt.subplots(1, 2, sharey=True,
#                              facecolor='w', figsize=(14, 9))

for io, field in enumerate(fields):
    idx = dfb['note'] == field
    sel = dfb[idx]
    rr = translate(sel)
    rr = rr.sort_values(by=[val])
    xmax = rr[val].max()
    idx = rr['season'] <= 6
    rra = rr[idx]
    ax.plot(rra[val], rra['numExposures'], marker=mk[field], color=colors[field],
            linestyle='None', mfc='None')


    # idx = rr['season'] >= 9
    # rrb = rr[idx]
    # ax2.plot(rrb[val], rrb['numExposures'], marker=mk[field], color=colors[field],
         linestyle='None', mfc='None')


    xmax = rr[val].max()
    # add seasons
    for seas in rra['season'].unique():
        ii = rra['season'] == seas
        selit = rra[ii]
        seas_min = selit[val].min()
        seas_max = selit[val].max()
        ax.plot([seas_min]*2, [0, ymax], color='k', linestyle='dotted')
        ax.plot([seas_max]*2, [0, ymax], color='k', linestyle='dotted')
        diff = seas_max-seas_min
        ax.text(seas_min+0.1*diff, 0.5*ymax,
                'Season', color='k', fontsize=15)
        ax.text(0.5*(seas_min+seas_max), 0.45*ymax,
                '{}'.format(int(seas)), color='k', fontsize=15)


# ax.set_ylim([0, ymax])
# ax.set_xlim([xmin, xmax])
ax.grid()
plt.show()
"""
