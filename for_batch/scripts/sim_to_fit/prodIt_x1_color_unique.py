#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:56:31 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import os

parser = OptionParser(
    description='Script to produce SN for WFD and DDF surveys')

parser.add_option("--runType", type="str", default='DDF',
                  help="type of run: DDF, WFD, DDF+WFD [%default]")
parser.add_option("--dbList_DD", type="str", default='DD_fbs_5.0.0.csv',
                  help="dbList DD to process  [%default]")
parser.add_option("--dbList_WFD", type="str", default='WFD_fbs_5.0.0.csv',
                  help="dbList WFD to process  [%default]")
parser.add_option("--outDir_main", type="str",
                  default='/sps/lsst/groups/cadence/LSST_SN_PhG/prod_simu',
                  help="Main output dir [%default]")
parser.add_option("--outDir_DD", type="str",
                  default='Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_faint',
                  help="output dir for DDF [%default]")
parser.add_option("--outDir_WFD", type="str",
                  default='Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_airmass_faint',
                  help="output dir for WFD [%default]")
parser.add_option("--SN_smearFlux", type=int,
                  default=1,
                  help="SN flux smearing [%default]")
parser.add_option("--Fitter_sigmaz", type=float,
                  default=1.e-05,
                  help="sigma_z for the fitter [%default]")
parser.add_option("--Observations_coadd", type=int,
                  default=1,
                  help="coadd observations [%default]")
parser.add_option("--saturation_effect", type=int,
                  default=0,
                  help="to include saturation effects [%default]")
parser.add_option("--SN_z_min", type=float,
                  default=0.01,
                  help="min redshift [%default]")
parser.add_option("--SN_z_max", type=float,
                  default=1.1,
                  help="max redshift [%default]")
parser.add_option("--SN_z_step", type=float,
                  default=0.1,
                  help="step redshift [%default]")
parser.add_option("--SN_x1_min", type=float,
                  default=-2.0,
                  help="x1 SN unique value [%default]")
parser.add_option("--SN_color_min", type=float,
                  default=0.2,
                  help="color SN unique value [%default]")
parser.add_option("--LC_coadd", type=int,
                  default=0,
                  help="coadd lc points [%default]")

opts, args = parser.parse_args()

params = vars(opts)

dbList_DD = ''
dbList_WFD = ''
outDir_DD = ''
outDir_WFD = ''

if 'DDF' in params['runType']:
    outDir_DD = '{}/{}'.format(params['outDir_main'], params['outDir_DD'])
    dbList_DD = params['dbList_DD']

if 'WFD' in params['runType']:
    outDir_WFD = '{}/{}'.format(params['outDir_main'],
                                str(params['outDir_WFD']))
    dbList_WFD = params['dbList_WFD']


script = 'python for_batch/scripts/sn_prod/prod_sn_dd_wfd.py'

cct = ['SN_smearFlux', 'Fitter_sigmaz',
       'Observations_coadd', 'saturation_effect','LC_coadd',
       'SN_z_min','SN_z_max','SN_z_step','SN_x1_min','SN_color_min']

nsn_fact_str = 'SN_NSNabsolute_DD'
if 'WFD' in params['runType']:
    nsn_fact_str = 'SN_NSNabsolute_WFD'
scr_ = script
scr_ += ' --outDir_DD={}'.format(outDir_DD)
scr_ += ' --dbList_DD={}'.format(dbList_DD)
scr_ += ' --outDir_WFD={}'.format(outDir_WFD)
scr_ += ' --dbList_WFD={}'.format(dbList_WFD)
scr_ += ' --{}=100'.format(nsn_fact_str)
scr_ += ' --SN_z_type=uniform'

for vv in cct:
    scr_ += ' --{}={}'.format(vv, params[vv])

print(scr_)
os.system(scr_)
