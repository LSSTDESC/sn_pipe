#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:59:56 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from sn_analysis.sn_selection import selection_criteria
from optparse import OptionParser
from sn_cosmology.fit_season import Fit_seasons
from sn_tools.sn_io import checkDir


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
parser.add_option("--outDir", type=str,
                  default='../cosmo_fit', help=" [%default]")

opts, args = parser.parse_args()

# loading SN data
dataDir_DD = opts.dataDir_DD
dbName_DD = opts.dbName_DD
dataDir_WFD = opts.dataDir_WFD
dbName_WFD = opts.dbName_WFD
selconfig = opts.selconfig
outDir = opts.outDir

checkDir(outDir)

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

outName = '{}/cosmo_{}.hdf5'.format(outDir, dbName_DD)
resfi = pd.DataFrame()

for key, vals in priors.items():
    cl = Fit_seasons(fitconfig, dataDir_DD, dbName_DD,
                     dataDir_WFD, dbName_WFD, dictsel, survey, vals)
    res = cl()
    res['prior'] = key
    resfi = pd.concat((resfi, res))
    """
    dict_fi = cl()
    for keyb, valb in dict_fi.items():
        dd = fitconfig[keyb]
        cosmopars = '_'.join(dd.keys())
        full_name = '{}_{}_{}.hdf5'.format(outName, cosmopars, key)
        valb.to_hdf(full_name, key='cosmo')
    """
resfi['dbName_DD'] == dbName_DD
resfi['dbName_WFD'] == dbName_WFD

resfi.to_hdf(outName, key='cosmo')
