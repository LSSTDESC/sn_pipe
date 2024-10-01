#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:56:03 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import glob
from optparse import OptionParser
import os

parser = OptionParser('script to transform npy effi(z) arry to cvs files')

parser.add_option("--dataDir", type=str,
                  default='../desi_desc_efficiencies_v2_with_desi2_strategies',
                  help="data dir[%default]")
parser.add_option("--outDir", type=str,
                  default='input/cosmology/host_effi',
                  help="output dir[%default]")
parser.add_option("--fileType", type=str,
                  default='host_efficiencies',
                  help="type of file (host_efficiencies/footprint)[%default]")
opts, args = parser.parse_args()

dataDir = opts.dataDir
outDir = opts.outDir
fileType = opts.fileType

fis = glob.glob('{}/*{}*.npy'.format(dataDir, fileType))

print('files', fis)
cmd = 'python run_scripts/cosmology/effi_to_csv.py'
cmd += ' --dataDir={}'.format(dataDir)
cmd += ' --outDir={}'.format(outDir)

for fi in fis:
    cmd_ = cmd
    fName = fi.split('/')[-1]
    fNameb = fName.split('{}_'.format(fileType))[-1]
    fNameOut = fNameb.split('.npy')[0]
    print(fName)
    cmd_ += ' --fName={}'.format(fName)
    cmd_ += ' --tagName={}'.format(fNameOut)
    print(cmd_)
    os.system(cmd_)
