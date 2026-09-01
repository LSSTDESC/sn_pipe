#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:23:53 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd


parser = OptionParser(description='Script to compare LCs on a large scale')

parser.add_option('--master_dirs', type=str, 
                  default='../prod_single/confe_z_0.81_1_0,../prod_single/confe_z_0.81_0_1',
                  help='master dirs for data [%default]')
parser.add_option('--dbName', type=str, default='baseline_v5.3.0_10yrs',
                  help='OS to process [%default]')
parser.add_option('--runType', type=str, default='DDF_spectroz',
                  help='run type [%default]')


opts, args = parser.parse_args()

master_dirs =opts.master_dirs.split(',')
dbName = opts.dbName
runType = opts.runType

master_a = master_dirs[0]
master_b = master_dirs[1]

dira = '{}/{}/{}'.format(master_a,dbName,runType)
dirb = '{}/{}/{}'.format(master_b,dbName,runType)

from sn_plotter_simu.visuLC import Comp_lc
Comp_lc(dira,dirb,todo='fit_all_diff')