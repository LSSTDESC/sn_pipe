#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:49:10 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sn_telmodel.sn_throughtools import Sigma_zp_meanwave
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config, add_parser
from sn_tools.sn_io import checkDir


def get_combi(dict_mean, dict_sigma, parList):
    """
    Function to build a df of combination of parameters

    Parameters
    ----------
    dict_mean : dict
        input dict with mean parameters.
    dict_sigma : dict
        input dict with sigma parameters.
    parList : str
        parameter list.

    Returns
    -------
    df : pandas df
        output data: all the possible parameter combinations.

    """

    dfa = get_df(dict_mean, parList)
    dfb = get_df(dict_sigma, parList)

    for vv in parList:
        dfb = dfb.rename(columns={vv: 'sigma_{}'.format(vv)})
    dfc = dfa.merge(dfb, how='cross')

    return dfc


def get_df(theDict, parList):
    """
    Build df from dict

    Parameters
    ----------
    theDict : dict
        parameter dict.
    parList : str
        parameter list.

    Returns
    -------
    df : pandas df
        output data.

    """

    df = pd.DataFrame()
    for vv in parList:
        dfa = pd.DataFrame(theDict[vv], columns=[vv])
        if len(df) > 0:
            df = df.merge(dfa, how='cross')
        else:
            df = dfa

    return df


# get all possible simulation parameters and put in a dict
path_input = 'input/zp_atmos'
confDict = make_dict_from_config(
    path_input, 'zp_atmos.txt')
parser = OptionParser(
    description='Script to estimate zp and mean_wave from atmos parameters')

add_parser(parser, confDict)

opts, args = parser.parse_args()

params = vars(opts)

# create outDir
checkDir(params['outDir'])

time_ref = time.time()
tel_dir = params['telDir']
through_dir = params['throughDir']
tag = params['tag']

tel_dir = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(tel_dir, through_dir)

site_name = 'LSST'
pressure = 743.
bands = 'ugrizy'

par_names = ['airmass', 'pwv', 'ozone', 'beta', 'aerosol']
"""
par_means = []
par_sigmas = []
for vv in par_names:
    par_means.append(params[vv])
    par_sigmas.append(params['sigma_{}'.format(vv)])
"""
dict_mean = {}
dict_sigma = {}
for vv in par_names:
    xmin = '{}_min'.format(vv)
    xmax = '{}_max'.format(vv)
    xstep = '{}_step'.format(vv)
    sigma_min = 'sigma_{}_min'.format(vv)
    sigma_max = 'sigma_{}_max'.format(vv)
    sigma_step = 'sigma_{}_step'.format(vv)
    vval = [params[xmin]]
    if params[xstep] > 1.e-8:
        vval = list(
            np.arange(params[xmin], params[xmax]+params[xstep], params[xstep]))
    dict_mean[vv] = vval
    sig = [params[sigma_min]]
    if params[sigma_step] > 1.e-8:
        sig = list(
            np.arange(params[sigma_min],
                      params[sigma_max]+params[sigma_step],
                      params[sigma_step]))
    dict_sigma[vv] = sig


combi_sigma = get_combi(dict_mean, dict_sigma, par_names)

df = pd.DataFrame()

print('nb de combinaisons', len(combi_sigma), combi_sigma.columns)
time_ref = time.time()
for i, row in combi_sigma.iterrows():

    time_ref_b = time.time()
    par_means = row[par_names].to_list()
    colsb = list(map(lambda x: 'sigma_' + x, par_names))
    par_sigmas = row[colsb].to_list()
    """
    print(par_names)
    print(par_means)
    print(par_sigmas)
    """
    sigma_zp = Sigma_zp_meanwave(through_dir, site_name, pressure,
                                 par_names, par_means, par_sigmas,
                                 save_throughputs_dir='ty_through')

    res = sigma_zp(ntrials=params['ntrial'], nproc=8)

    df = pd.concat((df, res))

    print('combi', time.time()-time_ref_b)
print('finally', time.time()-time_ref)
print('end of processing', time.time()-time_ref)

print(df[['std_zp_y', 'std_zp_z', 'std_mean_wave_y', 'std_mean_wave_z']])

outName = '{}/{}'.format(params['outDir'], params['outName'])

df.to_hdf(outName, key='zp_atmos')

"""
for b in bands:
    fig, ax = plt.subplots(figsize=(12, 9), ncols=2)
    ax[0].hist(df['std_zp_{}'.format(b)], histtype='step', bins=200)
    ax[1].hist(df['std_mean_wave_{}'.format(b)], histtype='step', bins=200)
    ax[0].set_xlabel(r'$\Delta z_p$')
    ax[1].set_xlabel(r'$\Delta \bar{\lambda}$')
    ax[0].set_ylabel(r'Number of Entries')
    ax[1].set_ylabel(r'Number of Entries')
"""

plt.show()
