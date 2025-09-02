#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:01:29 2025

@author: philippe.gris.clermont.in2p3.fr
"""
from optparse import OptionParser
import sn_phystools_input as cosmo_input
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
from sn_cosmology.random_hd import HD_random
import pandas as pd
import glob

# get all possible script parameters and put in a dict
path_cosmo_input = cosmo_input.__path__
confDict = make_dict_from_config(
    path_cosmo_input[0], 'sn_cosmology_survey.txt')

parser = OptionParser('script to fit cosmology parameters on SNe Ia surveys')

# parser for script parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

#grab params
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
priors = {}

if prior == 0:
    priors['noprior'] = pd.DataFrame()
else:

    priors['prior'] = pd.DataFrame({'varname': prior_varname,
                                    'refvalue': prior_refvalue,
                                    'sigma': prior_sigma})


dataDir = opts.dataDir
dbName_DD = opts.dbName_DD
dbName_WFD = opts.dbName_WFD

fitconfig = {}

fitconfig['fita'] = dict(zip(fitparams_names, fitparams_values))

#random instance
hd_random = HD_random(fitconfig=fitconfig)#,prior=priors)


# loop on data and make the fit

fis = glob.glob('{}/{}_{}/*.hdf5'.format(dataDir,dbName_DD,dbName_WFD))

print(len(fis))

n_random_surveys = int(len(fis)/10)

for nn in range(1,n_random_surveys+1):
    fis = glob.glob('{}/{}_{}/*_{}.hdf5'.format(dataDir,dbName_DD,dbName_WFD,nn))
    print(len(fis))
    sample_survey = pd.DataFrame()
    for fi in fis:
        rr = pd.read_hdf(fi)
        sample_survey = pd.concat((sample_survey,rr))
        print(len(sample_survey))
    #fit per year
    for year in range(1,11):
        idx = sample_survey['year'] <= year
        sel_sample = sample_survey[idx]
        vv = hd_random(sel_sample)
        print(vv)
    break
        
        
    
    
                                  
    
    