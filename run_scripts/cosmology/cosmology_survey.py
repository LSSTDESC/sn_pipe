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
from sn_cosmology.cosmo_tools import cosmo_dict,make_df
from sn_cosmology.random_hd import HD_random
from sn_cosmology.cosmo_tools import load_cosmo_params_from_script
from sn_tools.sn_utils import multiproc
import pandas as pd
import glob
import time


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
    prior = params['prior']

    cosmo_df = pd.DataFrame()
    print('processing', j, nreal)
    #nreal = [1]
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
            df_['dbName_DD'] = dbName_DD
            df_['dbName_WFD'] = dbName_WFD
            df_['prior'] = prior
            cosmo_df = pd.concat((cosmo_df, df_))

    if output_q is not None:
        return output_q.put({j: cosmo_df})
    else:
        return cosmo_df
 
def cosmo_tab_values(params):
    from sn_cosmology.cosmo_tabul import Cosmo_tabul
    from sn_tools.sn_interp import RegularGrid_interp
    # cosmo tabul
    costab = Cosmo_tabul(params)
    df_tot = costab()
    
    print(df_tot)
    #get interpolator
    ccols = params['cosmofitparams'].split(',')
    ccols.append('z')

    interpa = RegularGrid_interp(df_tot,ccols) 

    interp = interpa()
    
    return interp
    
    
# get all possible script parameters and put in a dict
path_cosmo_input = cosmo_input.__path__
confDict = make_dict_from_config(
    path_cosmo_input[0], 'sn_cosmology_survey.txt')

parser = OptionParser('script to fit cosmology parameters on SNe Ia surveys')

# parser for script parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

params = vars(opts)

fitparams_names = params['fitparam_names'].split(',')
fitparams_values = list(map(float, params['fitparam_values'].split(',')))

prior_refvalue = list(map(float, params['prior_refvalue'].split(',')))
prior_sigma = list(map(float, params['prior_sigma'].split(',')))

priors = pd.DataFrame()

if params['prior'] == 1:
    priors = pd.DataFrame({'varname': params['prior_varname'].split(','),
                           'refvalue': prior_refvalue,
                           'sigma': prior_sigma})
#load cosmology parameters and model

cosmodict = load_cosmo_params_from_script(params)

for key,val in cosmodict['de_params'].items():
    cosmodict[key] = val

#grab fitparams_values

dpar = {}
fitcosmo_params = []
for vv in fitparams_names:
    if vv != 'sigmaInt':
        dpar[vv] = cosmodict[vv]
        fitcosmo_params.append(vv)

if 'sigmaInt' in fitparams_names:
    dpar['sigmaInt'] = params['sigmaInt']

fitconfig = {}

#fitconfig['fita'] = dict(zip(fitparams_names, fitparams_values))

fitconfig['fita'] = dpar

params['cosmofitparams'] = ','.join(fitcosmo_params)
#grab distmod values
distmod_interp = None
if params['distmod_from_tabul']:
    distmod_interp = cosmo_tab_values(params)
    
# random HD

hd_random = HD_random(fitconfig=fitconfig, 
                      fitcosmo_params=fitcosmo_params,
                      cosmodict=cosmodict,
                      prior=priors,distmod_interp=distmod_interp)

# loop on data and make the fit

fis = glob.glob('{}/{}_{}/*.hdf5'.format(params['dataDir'], 
                                         params['dbName_DD'], 
                                         params['dbName_WFD']))

n_random_surveys = int(len(fis)/10)

n_real = list(range(1, n_random_surveys+1))

pp = {}
pp['dataDir'] = params['dataDir']
pp['dbName_DD'] = params['dbName_DD']
pp['dbName_WFD'] = params['dbName_WFD']
pp['yearmax'] = params['yearmax']
pp['hd_random'] = hd_random
pp['prior'] = params['prior']

time_ref = time.time()
if params['test_mode'] == 0:
    cosmo_df = multiproc(n_real, pp, cosmo_fits, nproc=params['nproc'])

    checkDir(params['outDir'])

    outName = '{}/cosmo_fit_{}_{}.hdf5'.format(params['outDir'], 
                                           params['dbName_DD'], 
                                           params['dbName_WFD'])

    cosmo_df.to_hdf(outName, key='cosmo')
else:
    cosmo_df = cosmo_fits([1],pp,0)
    cols = list(map(lambda st: '{}_fit'.format(st),fitparams_names))
    cols += ['Chi2_fit_red']
    print(cosmo_df[cols])

print('end fit cosmo', time.time()-time_ref)
