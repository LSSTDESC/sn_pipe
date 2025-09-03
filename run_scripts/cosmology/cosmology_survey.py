#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:01:29 2025

@author: philippe.gris.clermont.in2p3.fr
"""
from optparse import OptionParser
import sn_phystools_input as cosmo_input
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser, checkDir
from sn_cosmology.random_hd import HD_random
from sn_cosmology.cosmo_tools import transform
from sn_tools.sn_utils import multiproc
import pandas as pd
import glob
import time


def make_df(ddict):
    """
    Function to transform a dict to pandas df

    Parameters
    ----------
    ddict : dict
        Data to process.

    Returns
    -------
    res_df : pandas df
        output data.

    """

    res_df = pd.DataFrame()
    # fitted values in a df
    for key, vals in ddict.items():
        res = pd.DataFrame.from_dict(transform(vals))
        res['config'] = [key]
        res_df = pd.concat((res, res_df))

    return res_df


def cosmo_fits(nreal, params, j, output_q=None):
    """
    Function to perform cosmo_fits using multiprocessing

    Parameters
    ----------
    nreal : list(int)
        List of random surveys.
    params : dict
        parameters.
    j : int
        internal tag for multiprocessing.
    output_q : multiprocessing_queue, optional
        Where to put the results. The default is None.

    Returns
    -------
    pandas df
        Result.

    """

    dataDir = params['dataDir']
    dbName_DD = params['dbName_DD']
    dbName_WFD = params['dbName_WFD']
    yearmax = params['yearmax']
    hd_random = params['hd_random']

    cosmo_df = pd.DataFrame()
    print('processing', j, nreal)
    for nn in nreal:
        fis = glob.glob('{}/{}_{}/*_{}.hdf5'.format(dataDir,
                                                    dbName_DD, dbName_WFD, nn))
        sample_survey = pd.DataFrame()
        for fi in fis:
            rr = pd.read_hdf(fi)
            sample_survey = pd.concat((sample_survey, rr))
        # fit per year
        for year in range(1, yearmax+1):
            idx = sample_survey['year'] <= year
            sel_sample = sample_survey[idx]
            vv = hd_random(sel_sample)
            df_ = make_df(vv)
            df_['year'] = year+1
            df_['real_survey'] = nn
            cosmo_df = pd.concat((cosmo_df, df_))

    if output_q is not None:
        return output_q.put({j: cosmo_df})
    else:
        return cosmo_df


# get all possible script parameters and put in a dict
path_cosmo_input = cosmo_input.__path__
confDict = make_dict_from_config(
    path_cosmo_input[0], 'sn_cosmology_survey.txt')

parser = OptionParser('script to fit cosmology parameters on SNe Ia surveys')

# parser for script parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

# grab params
fitparams_names = opts.fitparam_names.split(',')
fitparams_values = list(map(float, opts.fitparam_values.split(',')))
prior = opts.prior
H0 = opts.H0
Om0 = opts.Om0
Ode0 = opts.Ode0
w0 = opts.w0
wa = opts.wa
alpha = opts.alpha
beta = opts.beta
recalc_sigmu = opts.recalc_sigmu
prior_varname = opts.prior_varname.split(',')
prior_refvalue = opts.prior_refvalue.split(',')
prior_sigma = opts.prior_sigma.split(',')

prior_refvalue = list(map(float, prior_refvalue))
prior_sigma = list(map(float, prior_sigma))

priors = pd.DataFrame()

if prior == 1:
    priors = pd.DataFrame({'varname': prior_varname,
                           'refvalue': prior_refvalue,
                           'sigma': prior_sigma})

dataDir = opts.dataDir
dbName_DD = opts.dbName_DD
dbName_WFD = opts.dbName_WFD
yearmax = opts.yearmax
nproc = opts.nproc
outDir = opts.outDir

fitconfig = {}

fitconfig['fita'] = dict(zip(fitparams_names, fitparams_values))

# random instance
hd_random = HD_random(fitconfig=fitconfig, prior=priors)


# loop on data and make the fit

fis = glob.glob('{}/{}_{}/*.hdf5'.format(dataDir, dbName_DD, dbName_WFD))

n_random_surveys = int(len(fis)/10)

n_real = list(range(1, n_random_surveys+1))

pp = {}
pp['dataDir'] = dataDir
pp['dbName_DD'] = dbName_DD
pp['dbName_WFD'] = dbName_WFD
pp['yearmax'] = yearmax
pp['hd_random'] = hd_random

time_ref = time.time()
cosmo_df = multiproc(n_real, pp, cosmo_fits, nproc=nproc)

checkDir(outDir)

outName = '{}/cosmo_fit_{}_{}.hdf5'.format(outDir, dbName_DD, dbName_WFD)

cosmo_df.to_hdf(outName, key='cosmo')

print('end fit cosmo', time.time()-time_ref)
