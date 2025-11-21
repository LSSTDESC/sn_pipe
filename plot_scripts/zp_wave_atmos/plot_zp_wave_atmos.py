#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 13:38:56 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
import glob
from optparse import OptionParser
from sn_tools.sn_utils import multiproc
from sn_plotter_analysis import plt


def process(fName):
    """
    Function to load and process a file

    Parameters
    ----------
    fName : str
        File name to process.

    Returns
    -------
    dfa: pandas df
       process data
    """

    # load the data
    df = pd.read_hdf(fName)

    df = df.round({'mean_airmass': 1})
    """
    print(df.columns)
    print(df['mean_airmass'].unique())
    """
    dfa = df.groupby(['mean_airmass']).apply(
        lambda x: stat(x), include_groups=False).reset_index()

    return dfa


def stat(grp):
    """
    Function to estimate some stats

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    dft : pandas df
        processed data.

    """

    str_ch = ['zp', 'mean_wave']
    bands = 'ugrizy'

    ccols = []

    for vva in str_ch:
        for b in bands:
            vvb = 'std_{}_{}'.format(vva, b)
            ccols.append(vvb)

    dfa = grp[ccols].mean().to_frame().T
    dfa.columns = dfa.columns.str.replace("std", "delta", regex=True)

    dfb = grp[ccols].std().to_frame().T
    dfb.columns = dfb.columns.str.replace("std", "std_delta", regex=True)

    dft = dfa.merge(dfb, how='cross')

    lparams = ['ozone', 'pwv', 'aerosol', 'beta']
    ccolsc = []
    for pps in ['mean', 'sigma']:
        for pp in lparams:
            ccolsc.append('{}_{}'.format(pps, pp))

    df_atm_param = grp[ccolsc].mean().to_frame().T

    dft = dft.merge(df_atm_param, how='cross')

    return dft


def process_multi(toproc, params, j=0, output_q=None):
    """
    Function to process data using multiprocessing

    Parameters
    ----------
    toproc : list(str)
        List of files to process.
    params : dict
        Parameter dict.
    j : int, optional
        tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to put the result. The default is None.

    Returns
    -------
    pandas df
        Output data.

    """

    res = pd.DataFrame()

    for pp in toproc:
        df = process(pp)
        res = pd.concat((df, res))

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


parser = OptionParser(
    description='Script to draw zp_sigma and mean_wave_eff vs atmos param errors')

parser.add_option('--fileDir', type=str, default='../zp_atmos',
                  help='file dir [%default]')

opts, args = parser.parse_args()

fileDir = opts.fileDir

fis = glob.glob('{}/*.hdf5'.format(fileDir))

params = {}
dft = multiproc(fis, params, process_multi, nproc=8)

print(dft)

fig, ax = plt.subplots()

print(dft.columns)
var = 'sigma_aerosol'
dft = dft.round({var: 5})
print(dft[var].unique())

sigmas_pwv = dft[var].unique()
markers = ['+', 'x', 'X', 's', 'P', '1', 'o']
mm = dict(zip(sigmas_pwv, markers))

for sig in sigmas_pwv:
    idx = dft[var] == sig
    sel = dft[idx]
    ax.plot(sel['mean_airmass'], sel['delta_zp_y'],
            marker=mm[sig], color='k', linestyle='None', label='{}'.format(sig))
ax.legend()
plt.show()
