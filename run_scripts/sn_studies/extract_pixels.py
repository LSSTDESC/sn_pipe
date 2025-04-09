#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:27:21 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import glob
from sn_tools.sn_io import checkDir

parser = OptionParser(description='Script to extract pixel infos')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA',
                  help='OS location dir [%default]')
parser.add_option('--dbList', type=str,
                  default='config_ana_selplot.csv',
                  help='list of OS to process [%default]')
parser.add_option('--fieldType', type=str,
                  default='DDF',
                  help='type of field to process [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='type of run [%default]')
parser.add_option('--outDir', type=str,
                  default='../pixels_DDF',
                  help='output Dir [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
fieldType = opts.fieldType
runType = opts.runType
outDir = opts.outDir

# create outDir if necessary
checkDir(outDir)

# load config
df_config = pd.read_csv(dbList, comment='#')

print(df_config)

cols = ['healpixID', 'pixRA', 'pixDec', 'season', 'season_length', 'field']
cols += ['survey_area', 'x1', 'color']

for i, row in df_config.iterrows():
    dbName = row['dbName']
    dirName = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, runType)
    df_res = pd.DataFrame()
    fis = glob.glob('{}/*.hdf5'.format(dirName))
    for fi in fis:
        df = pd.read_hdf(fi)
        df = df[cols]
        # df = df.drop_duplicates()
        df_res = pd.concat((df_res, df))
    print(len(df_res))
    # df_res = df_res.drop_duplicates()
    print(len(df_res))
    print(df_res)
    outName = '{}/{}.hdf5'.format(outDir, dbName)
    df_res.to_hdf(outName, key='pixels')
