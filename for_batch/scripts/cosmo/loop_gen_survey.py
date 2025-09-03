#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 13:32:18 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from optparse import OptionParser
from sn_tools.sn_batchutils import BatchIt
parser = OptionParser()

parser.add_option("--dataDir_DD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_G10_JLA',
                  help="db list to process  [%default]")
parser.add_option("--dataDir_WFD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_G10_JLA',
                  help="db list to process  [%default]")
parser.add_option("--surveyFile", type="str",
                  default='survey_scenario_spectroz_TiDES_5.csv',
                  help='survey file [%default]')
parser.add_option("--seasons", type="str",
                  default='1-10',
                  help='seasons/years to consider [%default]')
parser.add_option("--save_full_survey", type=int,
                  default=0,
                  help='to save the full survey or not [%default]')
parser.add_option("--n_random_survey", type=int,
                  default=50,
                  help='number of random surveys to generate [%default]')
parser.add_option("--select_WFD", int, default=1,
                  help='to select WFD SNe Ia')
parser.add_option("--surveyDir", str,
                  default='/sps/lsst/users/gris/sn_surveys',
                  help="output directory [%default]")
parser.add_option("--dbList", str,
                  default='',
                  help="list of db to process [%default]")

opts, args = parser.parse_args()

params = vars(opts)

del params['dbList']

script = 'run_scripts/cosmology/gen_survey.py'
"""
for key, vals in params.items():
    cmd_base += ' --{}={}'.format(key,vals)
"""
# grab list of dbs to process
dbNames = pd.read_csv(opts.dbList, comment='#')

# now build the batch
processName = 'gen_survey'
mybatch = BatchIt(processName=processName)

for i, row in dbNames.iterrows():
    params['dbName_DD'] = row['dbName']
    params['dbName_WFD'] = row['dbName']
    mybatch.add_batch(script, params)

mybatch.go_batch()
