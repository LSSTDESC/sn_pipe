#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:17:05 2025

@author: philippe.gris@clermont.in2p3.fr
"""

import pandas as pd
from optparse import OptionParser
from sn_tools.sn_batchutils import BatchIt

parser = OptionParser('script to fit (cosmo) LSST SN surveys')

parser.add_option("--dbList", type="str",
                  default='list_OS_new_wfd.csv',
                  help="db list to process [%default]")
parser.add_option("--outDir", type="str",
                  default='/sps/lsst/users/gris/cosmo_fit_last',
                  help="output directory [%default]")
parser.add_option("--dataDir", type="str",
                  default='/sps/lsst/users/gris/sn_surveys',
                  help="input directory for survey files[%default]")
parser.add_option('--fitparam_names', type=str,
                  default='w0,wa,Om0,sigmaInt',
                  help='fit parameter names [%default]')
parser.add_option('--fitparam_values', type=str,
                  default='-1.0,0.0,0.3,0.12',
                  help='fit parameter values [%default]')
parser.add_option('--prior', type=int,
                  default=1,
                  help='prior for the fit [%default]')
parser.add_option('--prior_varname', type=str, default='Om0',
                  help='prior varname list [%default]')
parser.add_option('--prior_refvalue', type=str, default='0.3',
                  help='prior refvalue list [%default]')
parser.add_option('--prior_sigma', type=str, default='0.0073',
                  help='prior sigma list [%default]')

opts, args = parser.parse_args()

dbList = opts.dbList
outDir = opts.outDir
dataDir = opts.dataDir
fitparam_names = opts.fitparam_names
fitparam_values = opts.fitparam_values
prior = opts.prior
prior_varname = opts.prior_varname
prior_refvalue = opts.prior_refvalue
prior_sigma = opts.prior_sigma


# load OS files to process
fis = pd.read_csv(dbList, comment='#')


script = 'run_scripts/cosmology/cosmology_survey.py'

pp = {}
pp['fitparam_names'] = fitparam_names
pp['fitparam_values'] = fitparam_values
pp['prior'] = prior
pp['prior_varname'] = prior_varname
pp['prior_refvalue'] = prior_refvalue
pp['prior_sigma'] = prior_sigma
pp['outDir'] = outDir
pp['dataDir'] = dataDir
# loop on files and create batches

for i, row in fis.iterrows():
    dbName = row['dbName']
    processName = 'cosmofit_survey_{}'.format(dbName)
    mybatch = BatchIt(processName=processName)
    pp['dbName_DD'] = dbName
    pp['dbName_WFD'] = dbName
    mybatch.add_batch(script, pp)
    mybatch.go_batch()
