#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 10:01:21 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import os

parser = OptionParser(description='Script to test cosmofit on multiple db')

parser.add_option('--dataDir_DD', type=str,
                  default='/sps/lsst/users/gris/Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA',
                  help='OS location dir for DDFs[%default]')
parser.add_option('--dataDir_WFD', type=str,
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA',
                  help='OS location dir for WFDs [%default]')
parser.add_option('--dbList', type=str,
                  default='config_cosmo.csv',
                  help='list of OS to process [%default]')
parser.add_option('--outDir', type=str,
                  default='/sps/lsst/users/gris/cosmo_fit_test',
                  help='output dir for cosmo results [%default]')
parser.add_option('--surveyDir', type=str,
                  default='/sps/lsst/users/gris/test_survey',
                  help='output dir for survey sn sample [%default]')

opts, args = parser.parse_args()

dataDir_DD = opts.dataDir_DD
dataDir_WFD = opts.dataDir_WFD
dbList = opts.dbList
outDir = opts.outDir
surveyDir = opts.surveyDir

df_list = pd.read_csv(dbList, comment='#')

cmd = 'python run_scripts/cosmology/test_cosmofit.py'
cmd += ' --dataDir_DD={}'.format(dataDir_DD)
cmd += ' --dataDir_WFD={}'.format(dataDir_WFD)
cmd += ' --survey=survey_scenario_spectroz_TiDES.csv'
cmd += ' --timescale=year'
cmd += ' --surveyDir={}'.format(surveyDir)
cmd += ' --outDir={}'.format(outDir)


for i, row in df_list.iterrows():
    dbName = row['dbName']
    cmd_ = cmd
    cmd_ += ' --dbName_DD={}'.format(dbName)
    cmd_ += ' --dbName_WFD={}'.format(dbName)
    cmd_ += ' --outName=cosmo_fit_{}'.format(dbName)
    print(cmd_)
    os.system(cmd_)
