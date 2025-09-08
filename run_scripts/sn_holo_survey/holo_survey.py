#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 11:07:25 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_holo_survey.holo_survey import HoloSurvey

parser = OptionParser(
    description='Script build an AuxTel survey')

parser.add_option('--dbDir', type=str,
                  default='../DB_Files',
                  help='Data dir [%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--nside', type=int,
                  default=128,
                  help='Healpix nside parameter [%default]')
parser.add_option('--deltaRA', type=float,
                  default=10,
                  help='RA width around pointing center [%default]')
parser.add_option('--deltaDec', type=float,
                  default=10,
                  help='Dec width around pointing center [%default]')
parser.add_option('--fp_level', type=str,
                  default='raft',
                  help='FP granularity level (ccd,raft,sensor) [%default]')
parser.add_option('--targetDir', type=str,
                  default='~/Bureau',
                  help='targets data dir [%default]')
parser.add_option('--targetFile', type=str,
                  default='gaia_source_file_ddf_v0.parquet',
                  help='targets data file [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='number of procs for multiprocessing [%default]')


opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
nside = opts.nside
deltaRA = opts.deltaRA
deltaDec = opts.deltaDec
fp_level = opts.fp_level
targetDir = opts.targetDir
targetFile = opts.targetFile
nproc = opts.nproc


myclass = HoloSurvey(nside,
                     deltaRA, deltaDec, fp_level,
                     targetDir, targetFile,
                     dbDir, dbName, nproc, show_Plot=True)

res = myclass()

print(res)
