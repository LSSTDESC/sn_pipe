#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:58:22 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import pandas as pd
from sn_analysis.sn_selection import selection_criteria, select
from sn_analysis.sn_tools import complete_df, get_pulls, sel_for_pull
import numpy as np
import re
import operator
import matplotlib.pyplot as plt


def load_data(dbDir, dbName, runType, field):
    """
    Function to load the data

    Parameters
    ----------
    dbDir : str
        data dir.
    dbName : str
        OS to process.
    runType : str
        runtype.
    field : str
        field.

    Returns
    -------
    df : pandas df
        loaded data.

    """

    theDir = '{}/{}/{}'.format(dbDir, dbName, runType)

    print('scanning', theDir)
    fis = glob.glob('{}/*{}*.hdf5'.format(theDir, field))

    df = pd.DataFrame()

    for fi in fis:

        df_ = pd.read_hdf(fi)

        df = pd.concat((df, df_))

    return df


def get_nsn(vala, valb, norm_factor):
    """
    Function to grab infos

    Parameters
    ----------
    vala : int
        number of sn after sel.
    valb : int
        number of sn before sel.
    norm_factor : float
        normalization factor.

    Returns
    -------
    list
        DESCRIPTION.

    """

    effi = vala/valb

    err_effi = np.sqrt(effi*(1.-effi)/valb)

    nsn = int(effi*valb/norm_factor)

    err_nsn = int(err_effi*valb/norm_factor)

    effi *= 100.
    err_effi *= 100.

    res = [(nsn, err_nsn, np.round(effi, 1), np.round(err_effi, 1))]
    cols = ['nsn', 'err_nsn', 'effi', 'err_effi']

    return pd.DataFrame(res, columns=cols)


def select_str(res, list_sel):
    """
    Function to select a pandas df

    Parameters
    ----------
    res : pandas df
        data to select.

    Returns
    -------
    pandas df
        selected df.

    """
    idx = True
    for vals in list_sel:
        idx &= vals[1](res[vals[0]], vals[2])
        mystr = '{} {} {}'.format(
            vals[0], get_symbol(vals[1].__doc__), vals[2])

    return mystr, res[idx]


def get_symbol(opdoc):
    """
    function to estimate symbol from operator.__doc__

    Parameters
    ----------
    opdoc : str
        operator.__doc__.

    Returns
    -------
    sym : str
        corresponding sym.

    """
    # sym = re.sub(r'.*\w\s?(\S+)\s?\w.*', '\\1', getattr(operator, op).__doc__)
    sym = re.sub(r'.*\w\s?(\S+)\s?\w.*', '\\1', opdoc)
    if re.match('^\\W+$', sym):
        return sym


def get_pull(sel, pullvar, fitgauss=True):

    fig, ax = plt.subplots()
    figtitle = pullvar
    fig.suptitle(pullvar)
    print('fitting', pullvar, sel[pullvar])

    # selb = pd.DataFrame(sel)
    selb = sel_for_pull(sel, pullvar, nstd=3.)

    ax.hist(selb[pullvar], histtype='step', bins=50)

    # Get the fitted curve
    if fitgauss:
        coeff = fit_pull(selb, pullvar)
        xmin = selb[pullvar].min()
        xmax = selb[pullvar].max()
        newbins = np.arange(xmin, xmax, 0.01)
        hist_fit = gauss(newbins, *coeff)
        mean = np.round(coeff[1], 2)
        sigma = np.round(coeff[2], 2)
        leg = 'pull= {} +- {}'.format(mean, sigma)
        ax.plot(newbins, hist_fit, label=leg)
        print('bbb', coeff[0], coeff[1], coeff[2])
    print(figtitle, np.mean(selb[pullvar]), np.std(selb[pullvar]))

    ax.grid(visible=True)
    ax.legend()

    # plt.show()


parser = OptionParser(description='Script to analyze SN selection criteria')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale [%default]')
parser.add_option('--seasons', type=str,
                  default='1',
                  help='seasons/years to process [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')
parser.add_option('--norm_factor', type=float,
                  default=30.,
                  help='normalization factor [%default]')
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help="sel config name[%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
runType = opts.runType
timescale = opts.timescale
seasons = opts.seasons.split(',')
seasons = list(map(int, seasons))
fields = opts.fields.split(',')
norm_factor = opts.norm_factor
selconfig = opts.selconfig

# selection vriteria
sellist = selection_criteria()[selconfig]

print(sellist)
rb = []
dfa = pd.DataFrame()
dfb = pd.DataFrame()

for field in fields:
    data = load_data(dbDir, dbName, runType, field)
    data = complete_df(data)
    print(field, len(data), len(data)/norm_factor)

    seasons = data['season'].unique()

    for seas in seasons:
        idx = data['season'] == seas
        mysel = data[idx]
        print('no sel', len(mysel)/norm_factor)
        n_nosel = int(len(mysel)/norm_factor)
        ra = get_pulls(mysel)
        ra['sel_str'] = 'nosel'
        ra['field'] = field
        ra['season'] = seas
        dfa = pd.concat((dfa, ra))
        ro = get_nsn(len(mysel), len(mysel), norm_factor)
        ro['sel_str'] = 'nosel'
        ro['field'] = field
        ro['season'] = seas
        dfb = pd.concat((dfb, ro))
        # get_pulls(mysel)
        for i in range(1, len(sellist)+1):
            ro = [field, int(seas)]
            mystr, sel = select_str(mysel, sellist[:i])
            ra = get_pulls(sel)
            ra['sel_str'] = mystr
            ra['field'] = field
            ra['season'] = seas
            dfa = pd.concat((dfa, ra))
            ro = get_nsn(len(sel), len(mysel), norm_factor)
            ro['sel_str'] = mystr
            ro['field'] = field
            ro['season'] = seas
            dfb = pd.concat((dfb, ro))

df_effi = dfa.merge(dfb,
                    left_on=['field', 'season', 'sel_str'],
                    right_on=['field', 'season', 'sel_str'],
                    suffixes=['', ''])

print(df_effi)
