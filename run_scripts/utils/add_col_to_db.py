#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:20:09 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import numpy as np
import numpy.lib.recfunctions as rf
import numpy._core.defchararray as np_f
from sn_tools.sn_io import checkDir

parser = OptionParser(description='analyze and plot of LC flux files')

parser.add_option('--inputDir', type=str, default='../DB_Files',
                  help='data input dir [%default]')
parser.add_option('--outputDir', type=str, default='../DB_Files',
                  help='data output dir [%default]')
parser.add_option('--dbName_in', type=str, 
                  default='baseline_v1.5_10yrs_orig.npy',
                  help='input os [%default]')
parser.add_option('--dbName_out', type=str, 
                  default='baseline_v1.5_10yrs.npy',
                  help='output os [%default]')

opts, args = parser.parse_args()

inputDir = opts.inputDir
outputDir = opts.outputDir
dbName_in = opts.dbName_in
dbName_out = opts.dbName_out

#dbName = 'baseline_v3.0_10yrs_orig.npy'
#outName =  'baseline_v3.0_10yrs.npy'

in_path = '{}/{}'.format(inputDir,dbName_in)
tt = np.load(in_path)

print('data loaded',len(tt))
tt = rf.append_fields(tt, ['scheduler_note','filter'], [tt['note'].tolist(),tt['band'].tolist()])
tt = rf.rename_fields(tt, {'Ra': 'RA'})

tt['scheduler_note'] = np_f.replace(tt['scheduler_note'],'EDFS','EDFS_a')

print(np.unique(tt['scheduler_note']))

checkDir(outputDir)
out_path = '{}/{}'.format(outputDir,dbName_out)
np.save(out_path,np.copy(tt))
