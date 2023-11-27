#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:50:21 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import os
from optparse import OptionParser
from sn_tools.sn_io import check_get_file

parser = OptionParser(description='Script to simulate light curves')

parser.add_option('--dbName', type=str, default='baseline_v3.2_10yrs_COSMOS',
                  help='dbName to process [%default]')
parser.add_option('--dbDir', type=str, default='../DB_Files',
                  help='dbDir of the OS to process [%default]')
parser.add_option('--tag', type=str, default='1.5',
                  help='tag versions of the throughputs [%default]')
parser.add_option('--WebPathSimu', type=str,
                  default='https://me.lsst.eu/gris/DESC_SN_pipeline',
                  help='web path for reference files [%default]')


opts, args = parser.parse_args()

dbName = opts.dbName
tag = opts.tag
dbDir = opts.dbDir
web = opts.WebPathSimu

# grab the file if necessary
fulldbName = '{}.npy'.format(dbName)
ffile = '{}/{}'.format(dbDir, fulldbName)
if not os.path.isfile(ffile):
    # if this file does not exist, grab it from a web server
    check_get_file(web, 'unittests', fulldbName, dbDir)

zpfile = 'zp_airmass_v{}.npy'.format(tag)
cmd = 'python run_scripts/simulation/run_simulation.py'
cmd += ' --dbName {}'.format(dbName)
cmd += ' --dbDir {} --dbExtens npy --OutputSimu_save 1'.format(dbDir)
cmd += ' --OutputSimu_throwafterdump 0 --SN_x1_type=unique --SN_x1_min=0.0'
cmd += ' --SN_color_type=unique --SN_color_min=0.0 --SN_z_type=uniform'
cmd += ' --SN_z_min 0.01 --SN_z_max 1.1 --SN_daymax_type=unique'
# cmd += ' --SN_z_type unique'
cmd += ' --Simulator_model salt3 --Simulator_version 2.0'
cmd += ' --MultiprocessingSimu_nproc 6'
cmd += ' --nproc_pixels 8 --nproc 1'
cmd += ' --SN_NSNfactor 1 --SN_NSNabsolute=1'
cmd += ' --ebvofMW_pixel 0.25 --OutputSimu_directory'
cmd += ' ../Output_SN_test_new/{}/DDF_spectroz'.format(dbName)
cmd += ' --fieldType DD --fieldName COSMOS'
cmd += ' --ProductionIDSimu SN_DD_{}_spectroz_1'.format(dbName)
cmd += ' --Observations_season 1 --nside 128 --Pixelisation_nside 128'
cmd += ' --npixels=1 --SN_sigmaInt=0.0'
cmd += ' --SN_nspectra=-1 --pixelList=pixelList.csv'
cmd += ' --InstrumentSimu_telescope_tag={}'.format(tag)

print(cmd)
os.system(cmd)
