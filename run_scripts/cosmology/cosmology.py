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


def host_effi_1D(lista, listb):
    """
    Function to build a dict of 1D interpolators

    Parameters
    ----------
    lista : list(str)
        List of csv files with z,effi as columns.
    listb : list(str)
        List of keys for the output dict.

    Returns
    -------
    dict_out : dict
        Output data.

    """

    from scipy.interpolate import interp1d
    dict_out = {}
    for i, vv in enumerate(lista):
        dd = pd.read_csv(vv, comment='#')
        nn = listb[i]
        dict_out[nn] = interp1d(dd['z'], dd['effi'],
                                bounds_error=False, fill_value=0.)

    return dict_out


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
                  default='baseline_v3.0_10yrs',
                  help="db name [%default]")
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help="sel config [%default]")
parser.add_option("--outDir", type=str,
                  default='../cosmo_fit', help="output dir [%default]")
parser.add_option("--survey", type=str,
                  default='input/DESC_cohesive_strategy/survey_scenario.csv',
                  help=" survey to use[%default]")
parser.add_option("--host_effi_UD", type=str,
                  default='input/DESC_cohesive_strategy/host_effi_Subaru.csv',
                  help="host effi curve for UD fields [%default]")
parser.add_option("--host_effi_DD", type=str,
                  default='input/DESC_cohesive_strategy/host_effi_4Most.csv',
                  help="host effi curve for DD+WFD fields [%default]")
parser.add_option("--frac_WFD_low_sigmaC", type=float,
                  default=0.8,
                  help="fraction of WFD SN with low sigmaC [%default]")
parser.add_option("--max_sigmaC", type=float,
                  default=0.04,
                  help="Max sigmaC defining low sigmaC sample [%default]")
parser.add_option("--test_mode", type=int,
                  default=0,
                  help="To run the test mode of the program [%default]")
parser.add_option("--lowz_optimize", type=float,
                  default=0.1,
                  help="To maximize the lowz sample [%default]")

opts, args = parser.parse_args()

# loading SN data
dataDir_DD = opts.dataDir_DD
dbName_DD = opts.dbName_DD
dataDir_WFD = opts.dataDir_WFD
dbName_WFD = opts.dbName_WFD
selconfig = opts.selconfig
outDir = opts.outDir
survey_file = opts.survey
host_effi_UD = opts.host_effi_UD
host_effi_DD = opts.host_effi_DD
frac_WFD_low_sigmaC = opts.frac_WFD_low_sigmaC
max_sigmaC = opts.max_sigmaC
test_mode = opts.test_mode
lowz_optimize = opts.lowz_optimize

checkDir(outDir)

dictsel = selection_criteria()[selconfig]
survey = pd.read_csv(survey_file, comment='#')

# make interp1d for host_effi

host_effi = host_effi_1D([host_effi_UD, host_effi_DD], [
    'host_effi_UD', 'host_effi_DD'])


# save the survey in outDir
outName_survey = '{}/survey_{}.csv'.format(outDir, dbName_DD)
survey.to_csv(outName_survey)


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
                     dataDir_WFD, dbName_WFD, dictsel, survey,
                     vals, host_effi, frac_WFD_low_sigmaC,
                     max_sigmaC, test_mode, lowz_optimize)
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
resfi['dbName_DD'] = dbName_DD
resfi['dbName_WFD'] = dbName_WFD

if test_mode:
    print('final result')
    print(resfi)

resfi.to_hdf(outName, key='cosmo')
