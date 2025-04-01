#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:00:32 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from optparse import OptionParser
from sn_plotter_analysis.sn_analyser_ddf import get_val
from sn_plotter_analysis.sn_analyser_summary import process_DDF
from scipy.interpolate import interp1d
import numpy as np
from sn_analysis.sn_calc_plot import bin_it_effi


def get_zpercentiles(data, fields, timescale, timeslots):
    """
    Function to get zpercentiles - loops on fields and dbNames.

    Parameters
    ----------
    data : pandas df
        Data to process.
    fields : list(str)
        List of fields to process.
    timescale : str
        timescale to use (year/season).
    timeslots : list(int)
        List of time slots to process.

    Returns
    -------
    df : pandas df
        Results.

    """

    dbNames = data['dbName'].unique()

    df = pd.DataFrame()
    for field in fields:
        for dbName in dbNames:
            dfa = z_percent(data, field, dbName, timescale, timeslots)
            df = pd.concat((df, dfa))

    return df


def z_percent(data, field, dbName, timescale, timeslots):
    """
    Function to grab zpercentiles - loop on timeslots    

    Parameters
    ----------
    data : pandas df
        Data to process.
    field : str
        Field to process.
    dbName : str
        db to process.
    timescale : str
        Time scale (year/season).
    timeslots : list(int)
        Time slots.

    Returns
    -------
    df : pandas df
        Output data.

    """

    idx = data['field'] == field
    idx &= data['dbName'] == dbName

    sel = data[idx]

    df = pd.DataFrame()
    for timeslot in timeslots:

        idxb = sel[timescale] == timeslot
        selb = sel[idxb]
        dfa = calc_zpercent(selb, 'z', 'sigmaC', 0.04)
        dfa[timescale] = timeslot
        dfa['dbName'] = dbName
        dfa['field'] = field

        df = pd.concat((df, dfa))

    return df


def calc_zpercent(selb, xvar, yvar, yvar_cut, frac=[0.8, 0.9]):
    """
    Function to estimate zpercentiles

    Parameters
    ----------
    selb : pandas df
        Data to process.
    xvar : str
        x-axis var.
    yvar : str
        y-axis sel var.
    yvar_cut : float
        yvar selection cut.
    frac : list(float), optional
        frac for zpercentiles estimation. The default is [0.8, 0.9].

    Returns
    -------
    df : pandas df
        output data.

    """

    df = bin_it_effi(selb, xvar=xvar, yvar=yvar, yvar_cut=yvar_cut,
                     bins=np.arange(0.0, 1.24, 0.08))

    # ax.errorbar(df['z'], df['sigma_mu'], yerr=df['sigma_mu_std'])
    xnew = np.linspace(np.min(df[xvar]), np.max(df[xvar]), 100)

    spl = interp1d(df[xvar], df['effi'], bounds_error=False, fill_value=0.)
    spl_smooth = spl(xnew)

    r = []
    for refval in frac:
        bb = interp1d(spl_smooth, xnew, bounds_error=False, fill_value=0.)
        r.append((refval, bb(refval)))

    df = pd.DataFrame(r, columns=['frac', 'zlim'])

    return df


parser = OptionParser(description='Script to estimate z_0.8 and z_0.9')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS DD list[%default]')
parser.add_option('--norm_factor', type=int,
                  default=30,
                  help='normalization factor [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='run type  [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--timeslots', type=str,
                  default='1-10',
                  help='time slot (season or year) to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='data type [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
norm_factor = opts.norm_factor
runType = opts.runType
config = opts.config
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
fields = opts.fields.split(',')
# cumul = opts.cumul
# plot_moll = opts.plot_Mollweid

dataType = opts.dataType

# read config file
conf_df = pd.read_csv(config, comment='#')

# process data
ddf = process_DDF(conf_df, dataType, dbDir, runType,
                  timescale, timeslots, norm_factor)

res = get_zpercentiles(ddf, fields, timescale, timeslots)

res.to_hdf('zpercentiles_ddf.hdf5', key='zper')
