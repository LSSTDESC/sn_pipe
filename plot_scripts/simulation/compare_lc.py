#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:23:53 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
#import pandas as pd
from sn_tools.sn_io import checkDir

parser = OptionParser(description='Script to compare LCs on a large scale')

parser.add_option('--master_dir', type=str, 
                  default='../prod_lcnew',
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
parser.add_option('--z', type=str, default='0.01',
                  help='redshift [%default]')
parser.add_option('--outDir', type=str, default='../comp_lc',
                  help='main outdir [%default]')

opts, args = parser.parse_args()

master_dir = opts.master_dir
dbName = opts.dbName
runType = opts.runType
configs = opts.configs.split(',')
config_fit = opts.config_fit
config_coadd = opts.config_coadd
z = opts.z
outDir='{}_{}_{}'.format(opts.outDir,config_fit,config_coadd)

checkDir(outDir)

dira = '{}/lc{}_{}_{}_{}/{}/{}'.format(master_dir,z,
                                 configs[0],config_fit,config_coadd,
                                 dbName,runType)

dirb = '{}/lc{}_{}_{}_{}/{}/{}'.format(master_dir,z,
                                 configs[1],config_fit,config_coadd,
                                 dbName,runType)

from sn_plotter_simu.visuLC import Comp_lc
ro = Comp_lc(dira,dirb,todo='fit_all_diff').res_fit
ro['config_ref'] = configs[0]
ro['config_test'] = configs[1]

outName = 'comp_{}_{}_z_{}.hdf5'.format(configs[0],configs[1],z)

outName_t = '{}/{}'.format(outDir,outName)

ro.to_hdf(outName_t,key='comp')