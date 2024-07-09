#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:28:10 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import os
from sn_tools.sn_batchutils import BatchIt

parser = OptionParser(description='Script to get reddening from stilism')

parser.add_option('--input_dir', type=str,
                  default='/sps/lsst/users/gris/gaia_files',
                  help='data dir [%default]')
parser.add_option('--output_dir', type=str,
                  default='/sps/lsst/users/gris/gaia_files_reddening',
                  help='output data dir [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='number of procs [%default]')

opts, args = parser.parse_args()

input_dir = opts.input_dir
output_dir = opts.output_dir
nproc = opts.nproc

# get the list of files to process

fis = glob.glob('{}/*.hdf5'.format(input_dir))

print(fis)

# loop on files and launch batch for each
time = '10:00:00'
mem = '40G'

procDict = {}
procDict['input_dir'] = input_dir
procDict['output_dir'] = output_dir
procDict['nproc'] = nproc
scriptref = 'run_scripts/utils/get_reddening.py'

for fi in fis:
    fName = fi.split('/')[-1]
    filename, extension = os.path.splitext(fName)
    procName = 'redden_{}'.format(filename)

    mybatch = BatchIt(processName=procName, time=time, mem=mem)
    procDict['input_file'] = fName
    procDict['output_file'] = fName

    mybatch.add_batch(scriptref, procDict)
