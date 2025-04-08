#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:01:00 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import os
from sn_tools.sn_io import checkDir

parser = OptionParser(
    description='Script to produce movies of NSN vs year (Mollweide, WFD)')

parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS list [%default]')
parser.add_option('--figDir', type=str,
                  default='../sn_wfd',
                  help='main fig dir[%default]')
parser.add_option('--outDir', type=str,
                  default='../fbs_WFD',
                  help='output directory[%default]')
opts, args = parser.parse_args()

config = opts.config
figDir = opts.figDir
outDir = opts.outDir

# load dbs
df = pd.read_csv(config, comment='#')

# check and create dir
checkDir(outDir)

# loop and produce the movies

script = 'python run_scripts/utils/make_movie_from_png.py'

for i, row in df.iterrows():
    dbName = row['dbName']
    cmd = script
    cmd += ' --figDir={}/{}'.format(figDir, dbName)
    cmd += ' --prefix=nsn_year --rate=1'
    cmd += ' --outName={}'.format(dbName)
    cmd += ' --extens=png'
    cmd += ' --movieDir={}'.format(outDir)
    os.system(cmd)
