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


def prepare_dict(par_names, params):
    """
    Function to estimate the parameter space for means and sigmas

    Parameters
    ----------
    par_names : list(str)
        List of parameters to consider.
    params : dict
        script parameters.

    Returns
    -------
    dict_mean : dict
        dict of mean values.
    dict_sigma : dict
        dict of sigma values.

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

    return dict_mean, dict_sigma


def process_combi(combi_sigma, params, num_combi, outName,nproc=8):
    """
    Fonction to process a set of parameters

    Parameters
    ----------
    combi_sigma : pandas df
        list of sets of parameters.
    params : dict
        script parameters.
    num_combi : int
        combi number.
    nproc: int, optional
        number of procs

    Returns
    -------
    df : pandas df
        output data.

    """

    time_ref = time.time()
    df = pd.DataFrame()
    ncombi = len(combi_sigma)
    print('ncombi:',ncombi)
    for i, row in combi_sigma.iterrows():

        time_ref_b = time.time()
        par_means = row[par_names].to_list()
        colsb = list(map(lambda x: 'sigma_' + x, par_names))
        par_sigmas = row[colsb].to_list()

        airmass = np.round(row['airmass'], 1)
        param_outName = 'params_airmass_{}_{}'.format(airmass, num_combi)
        sigma_zp = Sigma_zp_meanwave(through_dir, site_name, pressure,
                                     par_names, par_means, par_sigmas,
                                     save_throughputs_dir='',
                                     param_outDir=params['param_outDir'],
                                     param_outName=param_outName,
                                     save_random_dir=params['save_random_dir'])
        
        res = sigma_zp(ntrials=params['nsample'], nproc=nproc)

        df = pd.concat((df, res))

        rat = np.round(100.*i/ncombi,1)
        deltat = time.time()-time_ref_b
        print('combi', np.round(deltat,2),'s',rat,"%")
    print('finally', time.time()-time_ref)
    df['num_combi'] = num_combi

    # dump in file
    store.append('zp_atmos', df)


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

dict_mean, dict_sigma = prepare_dict(par_names, params)

combi_sigma = get_combi(dict_mean, dict_sigma, par_names)

# loop on the number of trials
outName = '{}/{}'.format(params['outDir'], params['outName'])
store = pd.HDFStore(outName, 'w')
for i in range(params['ntrial']):
    process_combi(combi_sigma, params, i+1, store,params['nproc'])

store.close()
