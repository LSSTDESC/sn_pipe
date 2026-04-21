#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 13:32:18 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from optparse import OptionParser
from sn_tools.sn_batchutils import BatchIt
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
import sn_phystools_input as cosmo_input

# get all possible script parameters and put in a dict
path_cosmo_input = cosmo_input.__path__
confDict = make_dict_from_config(
    path_cosmo_input[0], 'sn_loop_survey.txt')

parser = OptionParser('script to generate multiple LSST SN surveys')

# parser for script parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

dbList = opts.dbList

params = vars(opts)
tagName = params['tagName']

del params['dbList']
del params['tagName']

script = 'run_scripts/cosmology/gen_survey.py'
"""
for key, vals in params.items():
    cmd_base += ' --{}={}'.format(key,vals)
"""
# grab list of dbs to process
dbNames = pd.read_csv(dbList, comment='#')

# now build the batch
processName = 'gen_survey_{}'.format(tagName)

mybatch = BatchIt(processName=processName)

for i, row in dbNames.iterrows():
    params['dbName_DD'] = row['dbName']
    params['dbName_WFD'] = row['dbName']
    mybatch.add_batch(script, params)

mybatch.go_batch()
