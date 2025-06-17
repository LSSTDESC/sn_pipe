#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:46:22 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser
from sn_plotter_os_info.ddf_visits_night import analyze_simu_exp
from sn_plotter_analysis.sn_analyser_tools import clean_level
import matplotlib.pyplot as plt
from sn_plotter_metrics.plot4metric import plot_vs_OS


def ana_seq(df, timescale='year'):

    ccols_m = ['target_name', timescale]
    ccols = ccols_m+['seq_tot']

    dfb = df.groupby(ccols)[ccols].apply(
        lambda x: get_nvisits(x)).reset_index()
    dfb = clean_level(dfb)

    bands = 'ugrizy'
    for b in bands:
        ccols = ccols_m+[b]
        ccob = 'nnights_{}'.format(b)
        dfe = df.groupby(ccols)[ccols].apply(
            lambda x: get_nvisits_band(x, b, ccob)).reset_index()

        dfe = clean_level(dfe)
        print(dfe)
        dfb = dfb.merge(dfe, left_on=ccols_m,
                        right_on=ccols_m, suffixes=['', ''])
        print(dfb)

    print(dfb)
    ccols = ccols_m+['night']

    dfc = df.groupby(ccols_m)[ccols].apply(
        lambda x: get_nnights(x)).reset_index()
    dfc = clean_level(dfc)
    dfd = dfb.merge(dfc, left_on=ccols_m, right_on=ccols_m, suffixes=['', ''])

    dfd['seq_frac'] = 100.*dfd['nnights']/dfd['nnights_year']

    return dfd


def get_nvisits(grp, thevar='nnights'):

    dd = {}

    dd[thevar] = [len(grp)]

    res = pd.DataFrame.from_dict(dd)

    return res


def get_nvisits_band(grp, thevar, thevar_name='nnights'):

    dd = {}

    dd[thevar_name] = [grp[thevar].mean()]

    res = pd.DataFrame.from_dict(dd)

    return res


def get_nnights(grp, thevar='nnights_year'):

    dd = {}
    nights = grp['night'].unique()
    dd[thevar] = [len(nights)]

    res = pd.DataFrame.from_dict(dd)

    return res


def plot_seq_frac(data, dbName, field='COSMOS', season=1,
                  what='seq_frac', legy='sequence fraction [%]'):

    idx = data['target_name'] == field
    idx &= data['year'] == season
    sel = data[idx]

    figtit = dbName
    figtit += '\n {} - year {}'.format(field, season)

    plot_vs_OS(sel, varx='seq_tot',
               vary=what,
               legy=legy,
               title=figtit, fig=None, ax=None,
               label='', color='k', marker='.', ls='solid', mfc='k', mec='k')

    selb = sel.sort_values(by=[what], ascending=False)
    print(selb[['seq_tot', what]][:2])

    plt.show()


parser = OptionParser(
    description='Script to analyse DDF visits on a nightly basis from pointings')

parser.add_option("--dbDir", type="str",
                  default='../ddf_visits_night',
                  help="file directory [%default]")
parser.add_option("--dbName", type="str",
                  default='desc_ddf_v4.2.1_10yrs',
                  help="OS name [%default]")

opts, args = parser.parse_args()

# Load parameters
dbDir = opts.dbDir
dbName = opts.dbName

fName = '{}/{}.hdf5'.format(dbDir, dbName)

data = pd.read_hdf(fName)

print(data)

ro = ana_seq(data)

print(ro.columns)
print(test)

plot_seq_frac(ro, dbName, field='DD:COSMOS', season=3)

idx = ro['y'] > 0
plot_seq_frac(ro[idx], dbName, field='DD:COSMOS', season=3,
              what='nnights', legy='Number of nights')
# analyze_simu_exp(data)
