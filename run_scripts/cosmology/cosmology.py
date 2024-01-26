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
import numpy as np


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
parser.add_option("--host_effi_photz", type=str,
                  default='input/DESC_cohesive_strategy/host_effi_photz.csv',
                  help="host effi for photo-z [%default]")
parser.add_option("--frac_WFD_low_sigmaC", type=float,
                  default=0.8,
                  help="fraction of WFD SN with low sigmaC [%default]")
parser.add_option("--max_sigmaC", type=float,
                  default=0.04,
                  help="Max sigmaC defining low sigmaC sample [%default]")
parser.add_option("--test_mode", type=int,
                  default=0,
                  help="To run the test mode of the program [%default]")
parser.add_option("--plot_test", type=int,
                  default=0,
                  help="To run the test mode of the program and plot\
                  the samples distrib.[%default]")
parser.add_option("--lowz_optimize", type=float,
                  default=0.1,
                  help="To maximize the lowz sample [%default]")
parser.add_option("--sigmaInt", type=float,
                  default=0.12,
                  help="sigmaInt for SN [%default]")
parser.add_option("--surveyDir", type=str,
                  default='',
                  help="to dump surveys on disk [%default]")
parser.add_option('--timescale', type=str, default='year',
                  help='timescale for the cosmology (year or season)[%default]')
parser.add_option('--fields_for_stat', type=str,
                  default='COSMOS,XMM-LSS,ELAISS1,CDFS,EDFSa,EDFSb',
                  help='field list for stat (fit) [%default]')
parser.add_option('--seasons_cosmo', type=str,
                  default='1-10',
                  help='Seasons to estimate cosmology params [%default]')
parser.add_option('--nrandom', type=int,
                  default=50,
                  help='number of random sample (per season/year) to generate [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='number of procs to use [%default]')

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
host_effi_photz = opts.host_effi_photz
frac_WFD_low_sigmaC = opts.frac_WFD_low_sigmaC
max_sigmaC = opts.max_sigmaC
test_mode = opts.test_mode
plot_test = opts.plot_test
lowz_optimize = opts.lowz_optimize
sigmaInt = opts.sigmaInt
surveyDir = opts.surveyDir
timescale = opts.timescale
fields_for_stat = opts.fields_for_stat.split(',')
seasons_cosmo = opts.seasons_cosmo
nrandom = opts.nrandom
nproc = opts.nproc

if '-' in seasons_cosmo:
    seas = seasons_cosmo.split('-')
    seas_min = int(seas[0])
    seas_max = int(seas[1])
    seasons_cosmo = list(range(seas_min, seas_max+1))
else:
    seas = seasons_cosmo.split(',')
    seasons_cosmo = list(map(int, seas))

checkDir(outDir)

if surveyDir != '':
    checkDir(surveyDir)

dictsel = selection_criteria()[selconfig]
survey = pd.read_csv(survey_file, comment='#')

# make interp1d for host_effi

host_effi = host_effi_1D([host_effi_UD, host_effi_DD, host_effi_photz], [
    'host_effi_UD', 'host_effi_DD', 'host_effi_photz'])


# save the survey in outDir
seas_min = np.min(seasons_cosmo)
seas_max = np.max(seasons_cosmo)
outName_survey = '{}/survey_{}_{}_{}.csv'.format(
    outDir, dbName_DD, seas_min, seas_max)
survey.to_csv(outName_survey)


fitconfig = {}

"""
fitconfig['fita'] = dict(zip(['w0', 'Om0', 'alpha', 'beta', 'Mb'],
                             [-1, 0.3, 0.13, 3.1, -19.08]))

"""
"""
fitconfig['fitb'] = dict(zip(['w0', 'wa', 'Om0', 'alpha', 'beta', 'Mb'],
                             [-1, 0.0, 0.3, 0.13, 3.1, -19.08]))
"""

# fitconfig['fitc'] = dict(zip(['sigmaInt'],
#                              [0.12]))


# fitconfig['fitc'] = dict(zip(['w0', 'Om0'],
#                             [-1, 0.3]))
fitconfig['fitd'] = dict(zip(['w0', 'wa', 'Om0'],
                             [-1, 0.0, 0.3]))

priors = {}

# priors['noprior'] = pd.DataFrame()

priors['prior'] = pd.DataFrame({'varname': ['Om0'],
                                'refvalue': [0.3],
                               'sigma': [0.0073]})

outName = '{}/cosmo_{}_{}_{}.hdf5'.format(outDir,
                                          dbName_DD, seas_min, seas_max)
resfi = pd.DataFrame()

cl = Fit_seasons(fitconfig, dataDir_DD, dbName_DD,
                 dataDir_WFD, dbName_WFD, dictsel, survey,
                 priors, host_effi, frac_WFD_low_sigmaC,
                 max_sigmaC, test_mode, plot_test, lowz_optimize,
                 sigmaInt, surveyDir, timescale, outName,
                 fields_for_stat=fields_for_stat,
                 seasons=seasons_cosmo, nrandom=nrandom, nproc=nproc)
res = cl()
