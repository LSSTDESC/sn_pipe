#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:33:37 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sn_analysis.sn_tools import complete_df
from sn_analysis.sn_calc_plot import bin_it_mean
from sn_tools.sn_utils import multiproc
import time


def load_nosel_deprecated(dbDir, dbName, spectroType, field):
    """
    Functon to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        Db name.
    spectroType : str
        spectro type.
    field : str
        Field to load.

    Returns
    -------
    res : pandas df
        Loaded data.

    """

    fDir = '{}/{}/{}'.format(dbDir, dbName, spectroType)

    fis = glob.glob('{}/*_{}*.hdf5'.format(fDir, field))

    res = pd.DataFrame()
    for fi in fis:
        print('loading', fi)
        df = pd.read_hdf(fi)
        res = pd.concat((res, df))

    return res


def load_data(dbDir, dbName, spectroType, season, timescale='year'):
    """
    Functon to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        Db name.
    spectroType : str
        spectro type.
    field : str
        Field to load.

    Returns
    -------
    res : pandas df
        Loaded data.

    """
    fDir = '{}/{}/{}'.format(dbDir, dbName, spectroType)

    toload = '{}/*_{}_{}.hdf5'.format(fDir, timescale, season)
    print('to load', toload)
    fis = glob.glob(toload)

    res = pd.DataFrame()
    for fi in fis:
        print('loading', fi)
        df = pd.read_hdf(fi)
        res = pd.concat((res, df))

    print('fields', res['field'].unique())
    # idx = res['field'] == field

    return res


def plot_data(dfa, dfb, varx, vary):
    """
    Function to plot data

    Parameters
    ----------
    dfa : pandas df
        first data sample.
    dfb : pandas df
        second data sample.
    varx : str
        x-axis variable.
    vary : str
        y-axis variable.

    Returns
    -------
    None.

    """

    seasons = dfa['season'].unique()

    cols = ['x1_fit', 'color_fit', 'mbfit', 'mu']

    bins = np.arange(0.2, 1.15, 0.05)
    for seas in seasons:
        idxa = dfa['season'] == seas
        idxb = dfb['season'] == seas

        sela = dfa[idxa]
        selb = dfb[idxb]

        data_bina = bin_it_mean(sela, xvar='z_fit', yvar='mu', bins=bins)
        data_binb = bin_it_mean(selb, xvar='z_fit', yvar='mu', bins=bins)

        print(data_bina)

        fig, ax = plt.subplots()
        """
        ax.plot(sela[varx], sela[vary], 'ko')
        ax.plot(selb[varx], selb[vary], 'r*')
        """
        ax.plot(data_bina[varx], data_bina[vary], 'ko')
        ax.plot(data_binb[varx], data_binb[vary], 'r*')
        plt.show()


def plot_delta_mu(data, fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    print(data.columns)
    ndeg = 8
    bins = np.arange(0.0, 1.01, 0.1)
    data_bin = bin_it_mean(data, xvar='z_fit', yvar='diff_mu', bins=bins)
    print(data_bin)

    ax.errorbar(data_bin['z_fit'], data_bin['diff_mu'],
                yerr=data_bin['diff_mu_sigma'], lineStyle=None)

    """
    w = 1./data_bin['diff_mu_sigma']

    z, V = np.polyfit(data_bin['z_fit'],
                      data_bin['diff_mu'], ndeg, w=w, cov='unscaled')
    print(z)
    print(np.sqrt(np.diag(V)))
    p = np.poly1d(z)
    diff = p(data_bin['z_fit'])-data_bin['diff_mu']
    sigma = data_bin['diff_mu_sigma']
    ndof = data_bin['size'].sum()-ndeg-1
    print('chi2', np.sum((diff/sigma)**2)/ndof)
    bins = np.arange(0., 1.01, 0.01)
    ax.plot(bins, p(bins))
    """
    # ax.grid()


def plot_delta_mu_pixels(data_zstep):

    print(len(data_zstep))

    col = 'healpixID'
    hpixes = data_zstep[col].unique()

    for hpix in hpixes:
        # idx = data_nozstep[col] == hpix
        idxs = data_zstep[col] == hpix

        # d_nozstep = data_nozstep[idx]
        d_zstep = data_zstep[idxs]

        print(hpix, len(d_zstep))
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_delta_mu(d_zstep, fig=fig, ax=ax)

        df_cp = d_zstep.copy()

        for vv in ['x1', 'color', 'mb']:
            df_cp['{}_fit'.format(vv)] = df_cp['{}_fit'.format(
                vv)]+df_cp['delta_{}_corr'.format(vv)]

        df_cp = complete_df(df_cp)

        plot_delta_mu(df_cp, fig=fig, ax=ax)

        plt.show()


def correct_bias_sn_params(data_survey, data_zstep, alpha, beta):

    data_zstep = pd.DataFrame(data_zstep)
    cols = ['x1', 'delta_x1',
            'color', 'delta_color',
            'mb', 'delta_mb',
            'z_fit',
            'sigma_mu', 'sigma_z', 'season']

    for vv in ['x1', 'color', 'mb']:
        data_zstep['delta_{}'.format(
            vv)] = data_zstep[vv]-data_zstep['{}_fit'.format(vv)]
    data_zstep['sigma_z'] = 1.e-3

    data_ref = data_zstep[cols]

    seasons = data_ref['season'].unique()
    idx = data_survey['season'].isin(seasons)

    print('aoo', len(data_survey))
    ddi = data_survey.copy()

    params = {}
    params['data'] = ddi
    params['data_ref'] = data_ref

    indexes = range(len(ddi))
    print('indexes', indexes)
    time_ref = time.time()
    dd = multiproc(indexes, params, get_params_multi, 8)
    print('finally', time.time()-time_ref)
    # print(rr['Cov_colorcolor'])
    print(dd.columns)
    # rr = complete_df(dd, alpha, beta)

    print(dd)

    dd.to_hdf('test_corr.hdf', key='sn_bias_corr')
    fig, ax = plt.subplots()

    plot_pull(dd, fig=fig, ax=ax)
    plot_pull(dd, xfit='x1_corr', fig=fig, ax=ax)

    plt.show()


def get_params_multi(idata, params, j=0, output_q=None):

    imin = np.min(idata)
    imax = np.max(idata)

    print('proc', j, imin, imax)

    data = params['data']
    data_ref = params['data_ref']

    dd = data[imin:imax+1].copy()

    cols = ['delta_x1_corr', 'delta_color_corr', 'delta_mb_corr']
    dd[cols] = dd.apply(
        lambda x: get_params(x, data_ref, alpha, beta), axis=1)

    if output_q is not None:
        return output_q.put({j: dd})
    else:
        return dd


def get_params(row, data, alpha, beta):

    x1_fit = row.x1_fit
    color_fit = row.color_fit
    mb_fit = row.mb_fit
    z_fit = row.z_fit

    data = pd.DataFrame(data)

    data['X2'] = alpha*(data['x1']-data['delta_x1']-x1_fit)
    data['X2'] -= beta*(data['color']-data['delta_color']-color_fit)
    data['X2'] += (data['mb']-data['delta_mb']-mb_fit)

    data['X2'] *= data['X2']
    data['X2'] /= (data['sigma_mu']**2)

    data['X2'] += (data['z_fit']-z_fit)**2/data['sigma_z']**2

    data = data.sort_values(by=['X2'])

    thesel = data[:1]
    delta_x1_corr = thesel['delta_x1'].values[0]
    delta_color_corr = thesel['delta_color'].values[0]
    delta_mb_corr = thesel['delta_mb'].values[0]

    return pd.Series([delta_x1_corr, delta_color_corr, delta_mb_corr])


def plot_pull(data, xref='x1', xfit='x1_fit', sigma='sigma_x1', fig=None, ax=None):

    if fig is None:
        fig, ax = plt.subplots()

    toplot = (data[xref]-data[xfit]/data[sigma])
    idx = toplot >= -5
    idx &= toplot <= 5

    ax.hist(toplot[idx], histtype='step', bins=20)

    print('res pull', toplot[idx].mean(), toplot[idx].std())


parser = OptionParser(
    description='Script to study the malmquist bias on SNe Ia')

parser.add_option('--dbDir_survey', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA',
                  help='OS location dir - no selection[%default]')
parser.add_option('--dbDir_zstep', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_zstep_G10_JLA',
                  help='OS location dir - with selection[%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS',
                  help='fields to process [%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='data base to process [%default]')
parser.add_option('--spectroType', type=str,
                  default='DDF_spectroz',
                  help='spectro type [%default]')
parser.add_option('--alpha', type=float,
                  default=0.13,
                  help='nuisance parameter [%default]')
parser.add_option('--beta', type=float,
                  default=3.1,
                  help='nuisance parameter [%default]')
parser.add_option('--Mb', type=float,
                  default=-19.1,
                  help='nuisance parameter [%default]')

opts, args = parser.parse_args()

dbDir_survey = opts.dbDir_survey
dbDir_zstep = opts.dbDir_zstep
fields = opts.fields.split(',')
dbName = opts.dbName
spectroType = opts.spectroType
alpha = opts.alpha
beta = opts.beta
Mb = opts.Mb

tt = pd.read_hdf('test_corr.hdf')

print(tt)
plot_delta_mu_pixels(tt)
print(test)


data_survey = load_data(dbDir_survey, dbName, spectroType, 1, timescale='year')
data_zstep = load_data(dbDir_zstep, dbName, spectroType, 1, timescale='season')
data_survey = complete_df(data_survey, alpha, beta, Mb)
data_zstep = complete_df(data_zstep, alpha, beta, Mb)

fields = data_survey['field'].unique()


for field in fields:

    idx = data_survey['field'] == field
    data_survey_f = data_survey[idx]
    idxb = data_zstep['field'] == field
    data_zstep_f = data_zstep[idxb]

    correct_bias_sn_params(data_survey_f, data_zstep_f, alpha, beta)

    break

"""
for field in fields:
    # data_nosel = load_nosel(dbDir_nosel, dbName, spectroType, field)
    # print(len(data_nosel), data_nosel.columns)

    # data_nosel = complete_df(data_nosel, alpha, beta, Mb)
    data_survey = load_data(dbDir_survey, dbName, spectroType,
                            1, field, timescale='year')
    data_zstep = load_data(dbDir_zstep, dbName, spectroType,
                           1, field, timescale='season')

 
    plot_delta_mu_pixels(data_survey)
    

    # plot_data(d_nosel, d_sel, varx='z_fit', vary='mu')
"""
