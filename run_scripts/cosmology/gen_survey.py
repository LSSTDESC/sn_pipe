#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:50:36 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
import sn_phystools_input as cosmo_input
from sn_tools.sn_io import checkDir
import numpy as np
import pandas as pd
from sn_cosmology.cosmo_tools import load_footprints, load_host_effi
from sn_cosmology.cosmo_tools import load_data_season, random_LSST
from sn_cosmology.cosmo_tools import clean_survey, dump_survey, analyze_survey
from sn_cosmology.cosmo_tools import get_seasons, dump_survey_season
from sn_cosmology.random_hd import Random_survey
from sn_tools.sn_utils import multiproc
from sn_cosmology.random_survey import Gen_Surveys
import time

# get all possible script parameters and put in a dict
path_cosmo_input = cosmo_input.__path__
confDict = make_dict_from_config(
    path_cosmo_input[0], 'sn_survey.txt')

parser = OptionParser('script to generate LSST SN surveys')

# parser for script parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

pp = vars(opts)

time_ref = time.time()
mysurvey = Gen_Surveys(pp)

mysurvey()

print('generation done', time.time()-time_ref)

""" old code
seasons = get_seasons(pp['seasons'])

checkDir(pp['surveyDir'])

print('seasons', seasons)

survey = pd.read_csv(pp['surveyFile'], comment='#')

print('Survey considered', survey['survey'].unique())

# load host_effi
host_effi = load_host_effi(pp['hosteffiDir'], survey['host_effi'].unique())

# load footprints
footprints = load_footprints(pp['footprintDir'])

# save the survey in outDir
seas_min = np.min(seasons)
seas_max = np.max(seasons)


# Load the data

dataDir = {}

dataDir['DDF'] = pp['dataDir_DD']
dataDir['WFD'] = pp['dataDir_WFD']

dbName = {}
dbName['DDF'] = pp['dbName_DD']
dbName['WFD'] = pp['dbName_WFD']

fieldTypes = np.unique(survey[['zType', 'fieldType']].to_records(index=False))

# sort to have spectroz first
fieldTypes = sorted(fieldTypes.tolist())[::-1]

# load the data

vardf = ['z_fit', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',
         'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
         'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu',
         'field', 'healpixID', 'year', 'Cov_t0t0', 'x0_fit',
         'Cov_x0x0', 'Cov_x0x1', 'Cov_x0color', 'x0', 'x1',
         'color', 'SNID', 'season_length', 'survey_area']

sn_simu_season = {}
for seas in seasons:
    # load the data corresponding to this seas
    sn_simu_season[seas] = load_data_season(
        fieldTypes, dataDir, dbName, seas,
        select_WFD=pp['select_WFD'],
        select_DDF=pp['select_DDF'],
        timescale=pp['timescale'], vardf=vardf)

print('data loaded')


# prepare for a random survey
rand_survey = Random_survey(survey,
                            footprints, pp['timescale'],
                            pp['sigmaInt'], host_effi,
                            H0=pp['H0'],
                            Om0=pp['Om0'],
                            Ode0=pp['Ode0'],
                            w0=pp['w0'],
                            wa=pp['wa'],
                            alpha=pp['alpha'],
                            beta=pp['beta'],
                            low_z_optimize=pp['low_z_optimize'],
                            plot_test=pp['plot_test'],
                            test_mode=pp['test_mode'])


# normalisation factors
simu_norm_factor = pd.read_csv(pp['simu_norm_factor'], comment='#')

sn_sample = pd.DataFrame()
# take a random sample for each season

sreal = 1

for seas in seasons:
    # res = self.build_sample(sn_simu_season[seas], seas)
    sn_simu_seas = sn_simu_season[seas]

    # make a realization of this survey
    rand_LSST = random_LSST(
        sn_simu_seas, simu_norm_factor, test_mode=pp['test_mode'])

    full_survey = make_survey(rand_LSST)

    analyze_survey(full_survey)

    # make a random survey for the season
    res, res_foot = rand_survey(rand_LSST, seas)

    # concat this
    sn_sample = pd.concat((sn_sample, res))

    # clean the survey to remove duplicate
    sn_sample = clean_survey(sn_sample)

    analyze_survey(sn_sample)

    # dump the sample
    year_min = sn_sample[pp['timescale']].min()
    year_max = sn_sample[pp['timescale']].max()
    dump_survey(sn_sample, year_min, year_max, sreal, pp['surveyDir'],
                pp['dbName_DD'], pp['dbName_WFD'])
    if pp['save_full_survey']:
        dump_survey(full_survey, year_min, year_max, sreal, pp['surveyDir'],
                    pp['dbName_DD'], pp['dbName_WFD'], add_str='_nospectroz')
"""
