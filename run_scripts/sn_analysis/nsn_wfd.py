#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 08:55:14 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_plotter_analysis.sn_analyser_wdf import process_WFD_singledb
from sn_tools.sn_utils import get_val
from sn_tools.sn_io import checkDir
import pandas as pd

parser = OptionParser(description='Script to estimate nsn for WFD')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='list_wfd.csv',
                  help='list of OS to process [%default]')
parser.add_option('--norm_factor', type=int,
                  default=10,
                  help='Normalization factor [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz_nosat',
                  help='run type  [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--timeslots', type=str,
                  default='1-10',
                  help='time slot (season or year) to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')
parser.add_option('--outDir', type=str,
                  default='../sn_wfd',
                  help='output dir [%default]')
parser.add_option('--outName', type=str,
                  default='nsn_wfd.hdf5',
                  help='output name [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
norm_factor = opts.norm_factor
runType = opts.runType
dbList = opts.dbList
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
dataType = opts.dataType
outDir = opts.outDir
outName = opts.outName

# check outputdir
checkDir(outDir)

conf = pd.read_csv(dbList, comment='#')

for i, row in conf.iterrows():
    dbName = row['dbName_WFD']
    # check outputdir
    fName = f'{outDir}/{dbName}/{outName}'

    checkDir(f'{outDir}/{dbName}')
    # load, process and save wfd data
    process_WFD_singledb(dbName, dataType, dbDir, runType,
                         timescale, timeslots, norm_factor, fName)
