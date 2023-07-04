#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:59:56 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from sn_analysis.sn_selection import selection_criteria
from optparse import OptionParser
import numpy as np
from sn_cosmology.random_hd import HD_random, Random_survey, analyze_data
import time


def run_sequence(fitconfig, dataDir_DD, dbName_DD,
                 dataDir_WFD, dbName_WFD, dictsel, survey, prior):
    dict_fi = {}
    for seas_max in range(2, 12):
        seasons = range(1, seas_max)

        dict_res = fit_seasons(seasons, fitconfig, dataDir_DD, dbName_DD,
                               dataDir_WFD, dbName_WFD, dictsel, survey, prior)
        keys = dict_res.keys()
        for key in keys:
            if key not in dict_fi.keys():
                dict_fi[key] = pd.DataFrame()
            dict_fi[key] = pd.concat((dict_fi[key], dict_res[key]))

    return dict_fi


def fit_seasons(seasons, fitconfig, dataDir_DD, dbName_DD, dataDir_WFD,
                dbName_WFD, dictsel, survey, prior):

    hd_fit = HD_random(fitconfig=fitconfig, prior=prior)

    dict_res = {}

    for i in range(1):

        time_ref = time.time()
        data = Random_survey(dataDir_DD, dbName_DD,
                             dataDir_WFD, dbName_WFD,
                             dictsel, seasons,
                             survey=survey).data

        print('nsn', len(data))
        dict_ana = analyze_data(data)
        dict_ana['season'] = np.max(seasons)+1
        print(dict_ana)

        res = hd_fit(data)

        for key, vals in res.items():
            vals.update(dict_ana)
            res = pd.DataFrame.from_dict(transform(vals))
            if key not in dict_res.keys():
                dict_res[key] = pd.DataFrame()
            dict_res[key] = pd.concat((res, dict_res[key]))

    print('sequence', time.time()-time_ref)

    return dict_res


def transform(dicta):
    """
    Function to transform a dict of var to a dict of list(var)

    Parameters
    ----------
    dicta : dict
        input dict.

    Returns
    -------
    dictb : dict
        output dict.

    """

    dictb = {}

    for key, vals in dicta.items():
        dictb[key] = [vals]

    return dictb


parser = OptionParser()

parser.add_option("--dataDir_DD", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help="data dir[%default]")
parser.add_option("--dbName_DD", type=str,
                  default='DDF_Univ_WZ', help="db name [%default]")
parser.add_option("--dataDir_WFD", type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help="data dir[%default]")
parser.add_option("--dbName_WFD", type=str,
                  default='draft_connected_v2.99_10yrs',
                  help="db name [%default]")
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help=" [%default]")

opts, args = parser.parse_args()

# loading SN data
dataDir_DD = opts.dataDir_DD
dbName_DD = opts.dbName_DD
dataDir_WFD = opts.dataDir_WFD
dbName_WFD = opts.dbName_WFD
selconfig = opts.selconfig


dictsel = selection_criteria()[selconfig]
survey = pd.read_csv(
    'input/DESC_cohesive_strategy/survey_scenario.csv', comment='#')

fitconfig = {}
"""
fitconfig['fita'] = dict(zip(['w0', 'Om0', 'alpha', 'beta', 'Mb'],
                             [-1, 0.3, 0.13, 3.1, -19.08]))
fitconfig['fitb'] = dict(zip(['w0', 'wa', 'Om0', 'alpha', 'beta', 'Mb'],
                             [-1, 0.0, 0.3, 0.13, 3.1, -19.08]))
"""

# fitconfig['fitc'] = dict(zip(['sigmaInt'],
#                              [0.12]))

fitconfig['fitc'] = dict(zip(['w0', 'Om0'],
                             [-1, 0.3]))
fitconfig['fitd'] = dict(zip(['w0', 'wa', 'Om0'],
                             [-1, 0.0, 0.3]))


priors = {}

priors['noprior'] = pd.DataFrame()
priors['prior'] = pd.DataFrame({'varname': ['Om0'],
                                'refvalue': [0.3],
                                'sigma': [0.0073]})
outName = 'cosmo_{}'.format(dbName_DD)
for key, vals in priors.items():
    dict_fi = run_sequence(fitconfig, dataDir_DD, dbName_DD,
                           dataDir_WFD, dbName_WFD, dictsel, survey, vals)
    for keyb, valb in dict_fi.items():
        cosmopars = '_'.join(valb[0])
        full_name = '{}_{}_{}.hdf5'.format(outName, cosmopars, key)
        valb.to_hdf(full_name, key='cosmo')
