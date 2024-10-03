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
from sn_cosmology.cosmo_tools import get_survey_nickname
from sn_tools.sn_io import checkDir
import numpy as np
import glob
from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse
from sn_tools.sn_io import add_parser
import sn_phystools_input as cosmo_input


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


def load_host_effi(dataDir, llist):
    """
    Function to load all effi(z) csv files in a dict

    Parameters
    ----------
    dataDir : str
        Data dir.

    Returns
    -------
    dictout : dict
        key=name; val=interp1d(z,effi).

    """

    dictout = {}
    for ll in llist:
        fName = '{}/{}.csv'.format(dataDir, ll)
        rr = host_effi_1D([fName], [ll])
        dictout.update(rr)

    return dictout


def load_host_effi_deprecated(dataDir):
    """
    Function to load all effi(z) csv files in a dict

    Parameters
    ----------
    dataDir : str
        Data dir.

    Returns
    -------
    dictout : dict
        key=name; val=interp1d(z,effi).

    """

    fis = glob.glob('{}/*.csv'.format(dataDir))

    print('alors', dataDir, fis)
    dictout = {}
    for fi in fis:
        nname = fi.split('_')[-1].split('.')[0]
        rr = host_effi_1D([fi], ['host_effi_{}'.format(nname)])
        dictout.update(rr)

    return dictout


def load_footprints(dataDir):
    """
    Function to load footprints

    Parameters
    ----------
    dataDir : str
        Location dir of the footprint files.

    Returns
    -------
    df : pd.Dataframe
        Footprints (two cols: footprint, healpixID).

    """

    df = pd.DataFrame()

    import glob
    fis = glob.glob('{}/*.hdf5'.format(dataDir))

    for fi in fis:
        dfa = pd.read_hdf(fi)
        df = pd.concat((df, dfa))

    return df


# get all possible script parameters and put in a dict
path_cosmo_input = cosmo_input.__path__
confDict = make_dict_from_config(
    path_cosmo_input[0], 'sn_cosmology.txt')

parser = OptionParser('script to fit cosmology parameters')

# parser for script parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

# loading SN data
dataDir_DD = opts.dataDir_DD
dbName_DD = opts.dbName_DD
dataDir_WFD = opts.dataDir_WFD
dbName_WFD = opts.dbName_WFD
selconfig = opts.selconfig
outDir = opts.outDir
outName = opts.outName
survey_file = opts.survey
# lookup_survey_file = opts.lookup_survey
host_effi_dir = opts.hosteffiDir
max_sigma_mu = opts.max_sigma_mu
test_mode = opts.test_mode
plot_test = opts.plot_test
low_z_optimize = opts.low_z_optimize
sigmaInt = opts.sigmaInt
surveyDir = opts.surveyDir
timescale = opts.timescale
fields_for_stat = opts.fields_for_stat.split(',')
simu_norm_factor = pd.read_csv(opts.simu_norm_factor, comment='#')
seasons_cosmo = opts.seasons_cosmo
nrandom = opts.nrandom
nproc = opts.nproc
footprintDir = opts.footprintDir
wfd_tagsurvey = opts.wfd_tagsurvey
dd_tagsurvey = opts.dd_tagsurvey
# dd_surveys = opts.DD_surveys.split(',')
# wfd_surveys = opts.WFD_surveys.split(',')
fitparams_names = opts.fitparam_names.split(',')
fitparams_values = list(map(float, opts.fitparam_values.split(',')))
prior = opts.prior

if '-' in seasons_cosmo:
    seas = seasons_cosmo.split('-')
    seas_min = int(seas[0])
    seas_max = int(seas[1])
    seasons_cosmo = list(range(seas_min, seas_max+1))
else:
    seas = seasons_cosmo.split(',')
    seasons_cosmo = list(map(int, seas))

checkDir(outDir)

if surveyDir != 'None':
    checkDir(surveyDir)
else:
    surveyDir = ''

# load lookup table for survey
"""
survey_table = pd.read_csv(lookup_survey_file)

wfd_tagsurvey, wfd_surveys = get_survey_nickname(
    wfd_tagsurvey, wfd_surveys, survey_table)

dd_tagsurvey, dd_surveys = get_survey_nickname(
    dd_tagsurvey, dd_surveys, survey_table)
"""

dictsel = selection_criteria()[selconfig]
survey = pd.read_csv(survey_file, comment='#')

"""
# process only required surveys
idxa = survey_init['survey'].isin(dd_surveys)
idxb = survey_init['survey'].isin(wfd_surveys)

survey = pd.DataFrame(survey_init[idxa])
survey = pd.concat((survey, survey_init[idxb]))
"""
print('Survey considered', survey['survey'].unique())

# load host_effi
host_effi = load_host_effi(host_effi_dir, survey['host_effi'].unique())

# load footprints
footprints = load_footprints(footprintDir)

# save the survey in outDir
seas_min = np.min(seasons_cosmo)
seas_max = np.max(seasons_cosmo)
"""
outName_survey = '{}/survey_{}_{}_{}_{}_{}.csv'.format(
    outDir, dbName_DD, seas_min, seas_max, dd_tagsurvey, wfd_tagsurvey)
"""
outName_survey = '{}/{}.csv'.format(outDir, outName)
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

"""
fitconfig['fitc'] = dict(zip(['w0', 'Om0'],
                             [-1, 0.3]))
"""
fitconfig['fita'] = dict(zip(fitparams_names, fitparams_values))
"""
fitconfig['fitd'] = dict(zip(['w0', 'wa', 'Om0'],
                             [-1, 0.0, 0.3]))
"""
priors = {}

if prior == 0:
    priors['noprior'] = pd.DataFrame()
else:

    priors['prior'] = pd.DataFrame({'varname': ['Om0'],
                                    'refvalue': [0.3],
                                    'sigma': [0.0073]})

"""
outName = '{}/cosmo_{}_{}_{}_{}_{}.hdf5'.format(outDir,
                                                dbName_DD,
                                                seas_min, seas_max,
                                                dd_tagsurvey, wfd_tagsurvey)
"""
outName_full = '{}/{}.hdf5'.format(outDir, outName)
resfi = pd.DataFrame()

cl = Fit_seasons(fitconfig, dataDir_DD, dbName_DD,
                 dataDir_WFD, dbName_WFD, dictsel, survey,
                 priors, host_effi, footprints, low_z_optimize,
                 max_sigma_mu, test_mode, plot_test,
                 sigmaInt, surveyDir, timescale, outName_full,
                 fields_for_stat=fields_for_stat,
                 simu_norm_factor=simu_norm_factor,
                 seasons=seasons_cosmo, nrandom=nrandom,
                 nproc=nproc, wfd_tagsurvey=wfd_tagsurvey,
                 dd_tagsurvey=dd_tagsurvey)
res = cl()
