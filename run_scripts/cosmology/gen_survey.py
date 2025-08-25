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


# get all possible script parameters and put in a dict
path_cosmo_input = cosmo_input.__path__
confDict = make_dict_from_config(
    path_cosmo_input[0], 'sn_survey.txt')

parser = OptionParser('script to generate LSST SN surveys')

# parser for script parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

pp = vars(opts)

if '-' in pp['seasons']:
    seas = pp['seasons'].split('-')
    seas_min = int(seas[0])
    seas_max = int(seas[1])
    seasons = list(range(seas_min, seas_max+1))
else:
    seas = pp['seasons'].split(',')
    seasons = list(map(int, seas))

checkDir(pp['surveyDir'])

print('seasons', seasons)

survey = pd.read_csv(pp['survey_file'], comment='#')

print('Survey considered', survey['survey'].unique())

# load host_effi
host_effi = load_host_effi(pp['host_effi_dir'], survey['host_effi'].unique())

# load footprints
footprints = load_footprints(pp['footprintDir'])

# save the survey in outDir
seas_min = np.min(seasons)
seas_max = np.max(seasons)

"""
outName_survey = '{}/{}.csv'.format(outDir, outName)
survey.to_csv(outName_survey)
"""
