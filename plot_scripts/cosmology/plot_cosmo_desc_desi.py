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
import numpy as np
from optparse import OptionParser


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

    idx = grp[var] <= 1.e10
    sel = grp[idx]
    mean = sel[var].mean()
    std = sel[var].std()

    return pd.DataFrame({'{}_mean'.format(var): [mean], '{}_std'.format(var): [std]})


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


def plot_smom_year(dfb):

    fig, ax = plt.subplots()

    surveys = dfb['survey'].unique()

    idx = dfb['year'] > 2
    dfb = dfb[idx]
    for survey in surveys:
        idx = dfb['survey'] == survey
        sel = dfb[idx]
        ax.plot(sel['year'], sel['MoM_ratio'])

    ax.grid(visible=True)


def plot_val(df):

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


def plot_summary(dfb, surveys='plot_surveys.csv',
                 y_var='MoM_ratio',
                 y_leg='$\\frac{SMoM^{survey}}{SMoM^{TiDES}}$',
                 ascending=True):

    plot_surveys = pd.read_csv(surveys, comment='#')
    print('aoo', plot_surveys)

    idx = dfb['year'] == 11
    selb = dfb[idx]
    selb = selb.sort_values(by=[y_var], ascending=ascending)

    dfs = get_surveys('desc_desi_surveys')

    selb = selb.merge(dfs, left_on=['survey'], right_on=['survey'])
    # df_tot = df.merge(dfs, left_on=['survey'], right_on=['survey'])
    idxb = selb['surveylist'].isin(plot_surveys['surveylist'].to_list())
    selb = selb[idxb]
    selb = selb.merge(plot_surveys, left_on=['surveylist'], right_on=[
                      'surveylist'], suffixes=['', ''])

    figb, axb = plt.subplots(figsize=(18, 9))
    figb.subplots_adjust(bottom=0.20)
    axb.plot(selb['nickname'], selb[y_var], color='r', lw=2)
    axb.fill_between(selb['nickname'], selb['{}_plus'.format(y_var)],
                     selb['{}_minus'.format(y_var)], color='yellow')
    plt.setp(axb.get_xticklabels(), rotation=30,
             ha="right", rotation_mode="anchor", fontsize=12)
    axb.grid(visible=True)

    axb.set_ylabel(r'{}'.format(y_leg))


def get_mean_std(df, var='WFD'):
    """
    Function to estimate mean and std for a var.

    Parameters
    ----------
    df : pandas df
        data to process.
    var : str, optional
        var of interest. The default is 'WFD'.

    Returns
    -------
    dfc : pandas df
        output data.

    """

    dfc = df.groupby(['survey', 'year']).apply(
        lambda x: mean_std(x, var)).reset_index()

    varmean = '{}_mean'.format(var)
    varstd = '{}_std'.format(var)

    dfc['{}_plus'.format(varmean)] = dfc[varmean]+dfc[varstd]
    dfc['{}_minus'.format(varmean)] = dfc[varmean]-dfc[varstd]

    return dfc


def get_ratio(df, var='MoM'):
    """
    Method to estimate the ratio of var w.r.t a reference config.

    Parameters
    ----------
    df : pandas df
        Data to process.
    var : str, optional
        Variable of interest. The default is 'MoM'.

    Returns
    -------
    dfb : pandas df
        Output data.

    """

    dfb = get_mean_std(df, var)
    # ref survey
    idx = dfb['survey'] == 'scen_0'
    ref_df = pd.DataFrame(dfb[idx])

    dfb = dfb.merge(ref_df, left_on=['year'], right_on=[
                    'year'], suffixes=['', '_ref'])

    print(dfb.columns)

    var_mean = '{}_mean'.format(var)
    var_mean_r = '{}_mean_ref'.format(var)
    var_std = '{}_std'.format(var)
    var_std_r = '{}_std_ref'.format(var)
    var_ratio = '{}_ratio'.format(var)
    var_ratio_p = '{}_ratio_plus'.format(var)
    var_ratio_m = '{}_ratio_minus'.format(var)
    var_ratio_std = '{}_ratio_std'.format(var)

    dfb[var_ratio] = dfb[var_mean]/dfb[var_mean_r]
    dfb[var_ratio_std] = (dfb[var_std]/dfb[var_mean_r])**2
    dfb[var_ratio_std] += (dfb[var_mean] *
                           dfb[var_std_r])**2/dfb[var_mean_r]**4
    dfb[var_ratio_std] = np.sqrt(dfb[var_ratio_std])
    dfb[var_ratio_p] = dfb[var_ratio]+dfb[var_ratio_std]
    dfb[var_ratio_m] = dfb[var_ratio]-dfb[var_ratio_std]

    return dfb


parser = OptionParser(
    description='Script to plot cosmo results for DESC+DESI scenarios')
parser.add_option('--file_dir', type=str,
                  default='../cosmo_fit_desc_desi_new',
                  help='dir for files[%default]')
opts, args = parser.parse_args()

theDir = opts.file_dir

fis = glob.glob('{}/*.hdf5'.format(theDir))

df = pd.DataFrame()
for fi in fis:
    dfa = load(fi)
    df = pd.concat((df, dfa))

# estimate SMoM
df = recalc(df)
df['sigma_w'] = 100.*np.sqrt(df['Cov_w0_w0_fit'])
df['sigma_Om'] = 100.*np.sqrt(df['Cov_Om0_Om0_fit'])

print(df.columns.to_list())

print(df['survey'].unique(), df['MoM'])

dfb = df.groupby(['survey', 'year']).apply(lambda x: mean_std(x)).reset_index()


print(dfb)

"""
dfb = get_ratio(df, 'MoM')
idxb = dfb['MoM_ratio'] > 3
print(dfb[idxb])

# plot_smom_year(dfb)

plot_summary(dfb, y_var='MoM_ratio')
"""
var = ['WFD', 'sigma_w', 'sigma_Om']
leg = ['$N_{SN}$', '$\\sigma_w$ [%]', '$\\sigma_{\Omega_m}$ [%]']
ascl = [True, False, False]

var_r = ['MoM', 'sigma_w', 'sigma_Om']
leg_r = ['$\\frac{SMoM^{survey}}{SMoM^{TiDES}}$',
         '$\\frac{\sigma_w^{survey}}{\sigma_w^{TiDES}}$',
         '$\\frac{\sigma_{\Omega_m}^{survey}}{\sigma_{\Omega_m}^{TiDES}}$']
ascl_r = [True, False, False]

pplots = dict(zip(var, leg))
asc = dict(zip(var, ascl))

for key, vals in pplots.items():
    dfc = get_mean_std(df, key)
    plot_summary(dfc, y_var='{}_mean'.format(
        key), y_leg=vals, ascending=asc[key])

pplots_r = dict(zip(var_r, leg_r))
asc_r = dict(zip(var_r, ascl_r))

for key, vals in pplots_r.items():
    dfc = get_ratio(df, key)
    plot_summary(dfc, y_var='{}_ratio'.format(
        key), y_leg=vals, ascending=asc_r[key])

plt.show()
