#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:11:44 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser  
from sn_analysis.sn_selection import selection_criteria 

parser = OptionParser()

parser.add_option('--run_mode', type=str, default='sn_lc',
                  help='run mode (comp_lc,comp_lc_sn,comp_sn) [%default]')
parser.add_option('--master_dirs', type=str, 
                  default='../prod_single/confe_z_0.81_1_0,../prod_single/confe_z_0.81_0_1',
                  help='master dirs for data [%default]')
parser.add_option('--dbName', type=str, default='baseline_v5.3.0_10yrs',
                  help='OS to process [%default]')
parser.add_option('--runType', type=str, default='DDF_spectroz',
                  help='run type [%default]')


opts, args = parser.parse_args()

run_mode = opts.run_mode
master_dirs =opts.master_dirs.split(',')
dbName = opts.dbName
runType = opts.runType

dira = '../test_LC_confe_nocoadd/baseline_v5.3.0_10yrs/DDF_spectroz/'
dirb = '../test_LC_confd_nocoadd/baseline_v5.3.0_10yrs/DDF_spectroz/'

dira = '../test_LC_confe_coadd_bef_smearing_grizy/baseline_v5.3.0_10yrs/DDF_spectroz/'
#dirb = '../test_LC_confd_coadd_before_smearing/baseline_v5.3.0_10yrs/DDF_spectroz/'
dirb = '../test_LC_confe_obscoadd_grizy/baseline_v5.3.0_10yrs/DDF_spectroz/'
#dirb = '../test_LC_confe_coadd_after_smearing/baseline_v5.3.0_10yrs/DDF_spectroz/'

master_a = master_dirs[0]
master_b = master_dirs[1]

#master_a = '../test_new_1_0'
#master_a = '../test_new_0_1'

dira = '{}/{}/{}'.format(master_a,dbName,runType)
dirb = '{}/{}/{}'.format(master_b,dbName,runType)

snFile_a = 'SN_SN_DD_baseline_v5.3.0_10yrs_-2.0_0.2_1.hdf5'
snFile_b = 'SN_SN_DD_baseline_v5.3.0_10yrs_-2.0_0.2_1.hdf5'

sellist = selection_criteria()['G10_JLA']
print(sellist)

"""
import operator as op
sellist.append(('chisq_red',op.le,20))
sellist.append(('sigma_t0',op.le,0.5))
"""

#compare lcs
if run_mode == 'comp_lc':
    from sn_plotter_simu.visuLC import Comp_lc
    Comp_lc(dira,dirb,todo='plot_lc_super')

#compare lc fits
if run_mode == 'comp_lc_sn':
    from sn_plotter_simu.visuLC import Comp_lc_sn
    Comp_lc_sn(dira,snFile_a, dirb,snFile_b, sellist)    

#compare sn
if run_mode == 'comp_sn':
    from sn_plotter_simu.visuLC import Comp_sn
    Comp_sn(dira, snFile_a, dirb, snFile_b, sellist)


