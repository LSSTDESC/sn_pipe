#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:49:10 2024

@author: philippe.gris@clermont.in2p3.fr
"""

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sn_telmodel.sn_throughtools import Sigma_zp_meanwave
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config, add_parser
from sn_tools.sn_io import checkDir
from sn_tools.sn_utils import multiproc

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

def process_combi(combis, pparams, j=0, output_q=None):
    """
    Fonction to process a set of parameters

    Parameters
    ----------
    combis : pandas df
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
    
    params = pparams['params']
    num_combi = pparams['num_combi']

    time_ref = time.time()
    df = pd.DataFrame()
    ncombi = len(combis)
    print('ncombi:',ncombi)
    for i, row in combis.iterrows():
        res = process_single_combi(row, params,i,num_combi,ncombi)
        df = pd.concat((df, res))

       
    print('finally', time.time()-time_ref)
    df['num_combi'] = num_combi


    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df


    # dump in file
    #store.append('zp_atmos', df)

def process_single_combi(row,params,icombi,num_combi,ncombi):
    """
    Functio to process a single combi

    Parameters
    ----------
    row : pandas df
        input data.
    params : dict
        parameters.
    icombi : int
        i for combi.
    num_combi : int
        num combi.
    ncombi : int
        total number of combi.

    Returns
    -------
    res : pandas df
        output data.

    """
    
    time_ref_b = time.time()
    par_means = row[par_names].to_list()
    colsb = list(map(lambda x: 'sigma_' + x, par_names))
    par_sigmas = row[colsb].to_list()
    
    colsc = list(map(lambda x: 'bias_' + x, par_names))
    par_bias = row[colsc].to_list()
    
    airmass = np.round(row['airmass'], 1)
    param_outName = 'params_airmass_{}_{}'.format(airmass, num_combi)
    sigma_zp = Sigma_zp_meanwave(through_dir, site_name, pressure,
                             par_names, par_means, par_sigmas,
                             par_bias,
                             save_throughputs_dir='',
                             param_outDir=params['param_outDir'],
                             param_outName=param_outName,
                             save_random_dir=params['save_random_dir'])
    if params['nsample'] == 1:
        nproc=1
    
    res = sigma_zp(ntrials=params['nsample'], nproc=nproc)    
    
    """
    rat = np.round(100.*icombi/ncombi,1)
    deltat = time.time()-time_ref_b
    print('combi', np.round(deltat,2),'s',rat,"%")
    """
    return res

def get_combi_bias(cols,params,prefix='bias'):
    """
    Function to estimate combinations of biases

    Parameters
    ----------
    cols : list(str)
        List of columns to consider.
    params : dict
        parameter values.
    prefix : str, optional
        prefix to use. The default is 'bias'.

    Returns
    -------
    df : pandas df
        combi result.

    """
    
    colsb_min = list(map(lambda x: '{}_{}_min'.format(prefix,x), cols))
    
    df = pd.DataFrame()
    for i,vv in enumerate(cols):
        col_min = '{}_{}_min'.format(prefix,vv)
        col_max = '{}_{}_max'.format(prefix,vv)
        col_step = '{}_{}_step'.format(prefix,vv)

        vals = [params[col_min]]
        if params[col_step] > 1.e-8:
            vals = np.arange(params[col_min],
                             params[col_max]+params[col_step],
                             params[col_step])
        
        ddf = pd.DataFrame(vals,columns=['{}_{}'.format(prefix,vv)])
        
        if i == 0:
            df = ddf.copy()
        else:
            df = df.merge(ddf,how='cross')


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

dict_mean, dict_sigma = prepare_dict(par_names, params)

combi_sigma = get_combi(dict_mean, dict_sigma, par_names)
combi_bias = get_combi_bias(par_names,params,prefix='bias')

#combine combis

combis = combi_sigma.merge(combi_bias,how='cross')
combis['num_combi'] = combis.reset_index().index

outName = '{}/{}'.format(params['outDir'], params['outName'])
store = pd.HDFStore(outName, 'w')


pparams = {}
pparams['params'] = params

# loop on the number of trials
for i in range(params['ntrial']):
    pparams['num_combi'] =+1
    if params['nsample'] == 1:
        res = multiproc(combis,pparams,process_combi,params['nproc'])
    else:
        res = process_combi(combis, pparams)

print(res)
store.append('zp_atmos', res)
store.close()
