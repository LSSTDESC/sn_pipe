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


def load_nosel(dbDir, dbName, spectroType, field):
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


def load_sel(dbDir, dbName, spectroType, field):
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

    fis = glob.glob('{}/*.hdf5'.format(fDir))

    res = pd.DataFrame()
    for fi in fis:
        print('loading', fi)
        df = pd.read_hdf(fi)
        res = pd.concat((res, df))

    idx = res['field'] == field

    return res[idx]


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


parser = OptionParser(
    description='Script to study the malmquist bias on SNe Ia')

parser.add_option('--dbDir_nosel', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot',
                  help='OS location dir - no selection[%default]')
parser.add_option('--dbDir_sel', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA',
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

dbDir_nosel = opts.dbDir_nosel
dbDir_sel = opts.dbDir_sel
fields = opts.fields.split(',')
dbName = opts.dbName
spectroType = opts.spectroType
alpha = opts.alpha
beta = opts.beta
Mb = opts.Mb

col = 'healpixID'


for field in fields:
    data_nosel = load_nosel(dbDir_nosel, dbName, spectroType, field)
    print(len(data_nosel), data_nosel.columns)

    data_nosel = complete_df(data_nosel, alpha, beta, Mb)
    data_sel = load_sel(dbDir_sel, dbName, spectroType, field)
    """
    data_nosel['mb_fit'] = -2.5*np.log10(data_nosel['x0_fit']) + 10.635
    data_nosel['mu'] = alpha*data_nosel['x1_fit']-beta * \
        data_nosel['color_fit']+data_nosel['mb_fit']-Mb
    """
    print(len(data_sel))

    hpixes = data_nosel[col].unique()

    for hpix in hpixes:
        idx = data_nosel[col] == hpix
        idxs = data_sel[col] == hpix

        d_nosel = data_nosel[idx]
        d_sel = data_sel[idxs]

        print(hpix, len(d_nosel), len(d_sel))

        plot_data(d_nosel, d_sel, varx='z_fit', vary='mu')
