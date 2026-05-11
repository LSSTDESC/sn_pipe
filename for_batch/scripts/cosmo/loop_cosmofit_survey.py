#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:17:05 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from optparse import OptionParser
from sn_tools.sn_batchutils import BatchIt
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
#import sn_phystools_input as cosmo_input
import copy

# get all possible script parameters and put in a dict
#path_cosmo_input = cosmo_input.__path__
confDict = make_dict_from_config(
    'for_batch/input/cosmofit', 'cosmo_loop.txt')

parser = OptionParser('script to fit (cosmo) LSST SN surveys')

# parser for script parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

pp = vars(opts)

# load OS files to process
fis = pd.read_csv(pp['dbList'], comment='#')

script = 'run_scripts/cosmology/cosmology_survey.py'

# loop on files and create batches

for i, row in fis.iterrows():
    ppb = copy.deepcopy(pp)
    dbName = row['dbName']
    processName = 'cosmofit_survey_{}_{}'.format(dbName,ppb['tagName'])
    mybatch = BatchIt(processName=processName)
    del ppb['tagName']
    del ppb['dbList']
    ppb['dbName_DD'] = dbName
    ppb['dbName_WFD'] = dbName
    mybatch.add_batch(script, ppb)
    mybatch.go_batch()
