#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:48:05 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import glob
from sn_analysis.sn_tools import recalc
import matplotlib.pyplot as plt


def load(fName):
    """
    Function to load fName

    Parameters
    ----------
    fName : str
        File name to load.

    Returns
    -------
    df : pandas df
        Data.

    """

    df = pd.read_hdf(fName)
    survey = fName.split('/')[-1].split('.hdf5')[0].split('cosmo_fit_')[-1]
    df['survey'] = survey
    return df


def mean_std(grp, var='MoM'):
    """
    Function to get mean and std

    Parameters
    ----------
    grp : pandas df
        Data to process.
    var : str, optional
        Variable of interest. The default is 'MoM'.

    Returns
    -------
    pandas df
        Mean and std corresponding to var..

    """

    idx = grp['MoM'] <= 1.e10
    sel = grp[idx]
    mean = sel[var].mean()
    std = sel[var].std()

    return pd.DataFrame({'MoM_mean': [mean], 'MoM_std': [std]})


def get_surveys(theDir):
    """
    Function to get the survey

    Parameters
    ----------
    theDir : str
        Data directory.

    Returns
    -------
    df : pandas df
        Data.

    """

    fis = glob.glob('{}/*.csv'.format(theDir))

    r = []
    for fi in fis:
        scen = fi.split('/')[-1].split('.csv')[0].split('survey_scenario_')[1]
        dfa = pd.read_csv(fi, comment='#')
        surveys = dfa['survey'].tolist()
        strip_list = list(map(lambda it: it.strip('WFD_'), surveys))

        survey = '/'.join(strip_list)
        print('allo', scen, survey)
        r.append([scen, survey])

    df = pd.DataFrame(r, columns=['survey', 'surveylist'])

    return df


theDir = '../cosmo_fit_desc_desi_new'

fis = glob.glob('{}/*.hdf5'.format(theDir))

df = pd.DataFrame()
for fi in fis:
    dfa = load(fi)
    df = pd.concat((df, dfa))

# estimate SMoM
df = recalc(df)
print(df['survey'].unique(), df['MoM'])

dfb = df.groupby(['survey', 'year']).apply(lambda x: mean_std(x)).reset_index()

print(dfb)

# ref survey
idx = dfb['survey'] == 'scen_0'
ref_df = pd.DataFrame(dfb[idx])

dfb = dfb.merge(ref_df, left_on=['year'], right_on=[
                'year'], suffixes=['', '_ref'])

dfb['MoM_ratio'] = dfb['MoM_mean']/dfb['MoM_mean_ref']
print(dfb)

idxb = dfb['MoM_ratio'] > 3
print(dfb[idxb])


fig, ax = plt.subplots()

surveys = dfb['survey'].unique()

idx = dfb['year'] > 2
dfb = dfb[idx]
for survey in surveys:
    idx = dfb['survey'] == survey
    sel = dfb[idx]
    ax.plot(sel['year'], sel['MoM_ratio'])

ax.grid(visible=True)

fig, ax = plt.subplots()
idx = df['survey'] == 'scen_0'
idx &= df['year'] == 2
# idx &= df['MoM'] <= 5000
sel = df[idx]

vvar = 'Cov_Om0_Om0_fit'
vvar = 'Cov_w0_w0_fit'
vvar = 'Cov_Om0_w0_fit'
vvar = 'MoM'
vvarb = 'WFD_TiDES'
ax.hist(sel[vvar], histtype='step', bins=100)
# ax.plot(sel[vvarb], sel[vvar], 'k.')

print(sel[vvar])
print(sel.columns.tolist())

idx = dfb['year'] == 11
selb = dfb[idx]
selb = selb.sort_values(by=['MoM_ratio'])

dfs = get_surveys('desc_desi_surveys')

selb = selb.merge(dfs, left_on=['survey'], right_on=['survey'])
df_tot = df.merge(dfs, left_on=['survey'], right_on=['survey'])

figb, axb = plt.subplots(figsize=(18, 9))
figb.subplots_adjust(bottom=0.20)
axb.plot(selb['surveylist'], selb['MoM_ratio'])
plt.setp(axb.get_xticklabels(), rotation=30,
         ha="right", rotation_mode="anchor", fontsize=8)
axb.grid(visible=True)

axb.set_ylabel(r'$\frac{SMoM^{survey}}{SMoM^{TiDES}}$')

figc, axc = plt.subplots()

print(df_tot.columns)
df_tot = df_tot.fillna(0)
axc.plot(df_tot['surveylist'], df_tot['WFD_TiDES'], 'ko')


plt.show()
