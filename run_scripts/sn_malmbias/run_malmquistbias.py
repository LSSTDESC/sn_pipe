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


def load_sel(dbDir, dbName, spectroType, season, field):
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

    fis = glob.glob('{}/*_season_{}.hdf5'.format(fDir, season))

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


def plot_delta_mu(data):

    print(data.columns)
    ndeg = 8
    bins = np.arange(0.01, 1.01, 0.01)
    data_bin = bin_it_mean(data, xvar='z_fit', yvar='diff_mu', bins=bins)
    print(data_bin)
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.errorbar(data_bin['z_fit'], data_bin['diff_mu'],
                yerr=data_bin['diff_mu_sigma'], lineStyle=None)

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

    ax.grid()
    plt.show()


parser = OptionParser(
    description='Script to study the malmquist bias on SNe Ia')

"""
parser.add_option('--dbDir_nosel', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot',
                  help='OS location dir - no selection[%default]')
"""
parser.add_option('--dbDir_sel', type=str,
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

# dbDir_nosel = opts.dbDir_nosel
dbDir_sel = opts.dbDir_sel
fields = opts.fields.split(',')
dbName = opts.dbName
spectroType = opts.spectroType
alpha = opts.alpha
beta = opts.beta
Mb = opts.Mb

col = 'healpixID'


for field in fields:
    # data_nosel = load_nosel(dbDir_nosel, dbName, spectroType, field)
    # print(len(data_nosel), data_nosel.columns)

    # data_nosel = complete_df(data_nosel, alpha, beta, Mb)
    data_sel = load_sel(dbDir_sel, dbName, spectroType, 1, field)
    """
    data_nosel['mb_fit'] = -2.5*np.log10(data_nosel['x0_fit']) + 10.635
    data_nosel['mu'] = alpha*data_nosel['x1_fit']-beta * \
        data_nosel['color_fit']+data_nosel['mb_fit']-Mb
    """
    print(len(data_sel))

    hpixes = data_sel[col].unique()

    for hpix in hpixes:
        # idx = data_nosel[col] == hpix
        idxs = data_sel[col] == hpix

        # d_nosel = data_nosel[idx]
        d_sel = data_sel[idxs]

        print(hpix, len(d_sel))
        plot_delta_mu(d_sel)

        # plot_data(d_nosel, d_sel, varx='z_fit', vary='mu')
