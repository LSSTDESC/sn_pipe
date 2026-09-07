#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:28:11 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import numpy as np
from sn_tools.sn_batchutils import BatchIt

parser = OptionParser(description='Script to process LC comparison')

parser.add_option('--master_dir', type=str, 
                  default='/sps/lsst/groups/cadence/LSST_SN_PhG/prod_lcnew',
                  help='master dir for data [%default]')
parser.add_option('--dbName', type=str, default='baseline_v5.3.0_10yrs',
                  help='OS to process [%default]')
parser.add_option('--runType', type=str, default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--configs', type=str, default='confa,conff',
                  help='configs [%default]')
parser.add_option('--config_fit', type=str, default='nofit',
                  help='fit config [%default]')
parser.add_option('--config_coadd', type=str, default='nocoadd',
                  help='coadd config [%default]')
parser.add_option('--outDir', type=str, default='/sps/lsst/users/gris/comp_lc',
                  help='main outdir [%default]')

opts, args = parser.parse_args()

"""
master_dir = opts.master_dir
dbName = opts.dbName
runType = opts.runType
configs = opts.configs
config_fit = opts.config_fit
config_coadd = opts.config_coadd
outDir = opts.outDir
"""

pp = vars(opts)
zmin = 0.01
zmax = 1.1
zstep = 0.01

zvals = np.arange(zmin,zmax+zstep,zstep)

script = '-W ignore run_scripts/simulation/compare_lc.py'

procName = 'comp_lc_{}_{}_{}'.format(pp['configs'].replace(',','_'),
                                        pp['config_fit'],pp['config_coadd'])

mybatch = BatchIt(processName=procName)


for z in zvals:
    pp['z'] = np.round(z,2)
    if z <= 1.1:
        mybatch.add_batch(script,pp)
    
mybatch.go_batch()