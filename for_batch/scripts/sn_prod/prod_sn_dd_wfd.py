#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:11:56 2023

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import os

parser = OptionParser()

parser.add_option("--dbList_DD", type="str", default='DD_fbs_3.3.csv',
                  help="db list to process (DDF) [%default]")
parser.add_option("--outDir_DD", type="str", default='/sps/lsst/users/gris/Output_SN_sigmaInt_0.0_Hounsell_new',
                  help="output dir for DDF prod [%default]")
parser.add_option("--dbList_WFD", type="str", default='WFD_fbs_3.3.csv',
                  help="db list to process (WFD) [%default]")
parser.add_option("--outDir_WFD", type="str",
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_new',
                  help="output dir for WFD prod [%default]")
parser.add_option("--SN_smearFlux", type=int, default=1,
                  help="LC flux smearing [%default]")
parser.add_option("--Fitter_sigmaz", type=float, default=1.e-5,
                  help="sigmaz for LC fits [%default]")
parser.add_option("--SN_sigmaz", type=float, default=1.e-5,
                  help="sigmaz for LC zsim [%default]")
parser.add_option("--simuParams_fromFile", type=int, default=0,
                  help="to use simulation params from file [%default]")
parser.add_option("--simuParams_dir_DD", type=str,
                  default='/sps/lsst/users/gris/Output_SN_sigmaInt_0.0_Hounsell_nnew',
                  help="Dir for the simu param files  - DD[%default]")
parser.add_option("--simuParams_dir_WFD", type=str,
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_nnew',
                  help="Dir for the simu param files  - WFD[%default]")
parser.add_option("--DD_list", type=str,
                  default='COSMOS,CDFS,EDFS,ELAISS1,XMM-LSS',
                  help="List of DDFs to process [%default]")
parser.add_option("--lookup_ddf", type=str,
                  default='input/simulation/lookup_ddf.csv',
                  help="Look up table for DDFs [%default]")
parser.add_option("--saturation_effect", type=int,
                  default=0,
                  help="to include saturation effects [%default]")
parser.add_option("--saturation_psf", type=str,
                  default='single_gauss',
                  help="psf to use for saturation effects [%default]")
parser.add_option("--saturation_ccdfullwell", type=float,
                  default=120000,
                  help="ccd full well to use for saturation effects [%default]")
parser.add_option("--SN_z_max", type=float,
                  default=1.1,
                  help="zmax for sn prod [%default]")
parser.add_option("--SN_NSNfactor_DDF", type=int,
                  default=30,
                  help="nsn factor for DDF production[%default]")
parser.add_option("--SN_NSNfactor_WFD", type=int,
                  default=10,
                  help="nsn factor for WFD production[%default]")
parser.add_option("--Observations_coadd", type=int,
                  default=1,
                  help="obs coadd [%default]")
parser.add_option("--fit_remove_sat", type=str,
                  default='0',
                  help="to remove LC saturated points [%default]")

opts, args = parser.parse_args()

dbList_DD = opts.dbList_DD
outDir_DD = opts.outDir_DD

dbList_WFD = opts.dbList_WFD
outDir_WFD = opts.outDir_WFD

smearFlux = opts.SN_smearFlux
sigmaz = opts.Fitter_sigmaz
simuParams_fromFile = opts.simuParams_fromFile
simuParams_dir_DD = opts.simuParams_dir_DD
simuParams_dir_WFD = opts.simuParams_dir_WFD
dd_list = opts.DD_list
lookup_ddf = opts.lookup_ddf
saturation_effect = opts.saturation_effect
saturation_psf = opts.saturation_psf
saturation_ccdfullwell = opts.saturation_ccdfullwell
sn_z_max = opts.SN_z_max
sn_nsn_factor_dd = opts.SN_NSNfactor_DDF
sn_nsn_factor_wfd = opts.SN_NSNfactor_WFD
obs_coadd = opts.Observations_coadd
sn_sigmaz = opts.SN_sigmaz
fit_remove_sat = opts.fit_remove_sat

cmd_scr = 'python for_batch/scripts/sn_prod/loop_prod.py'
cmd_scr += ' --SN_sigmaInt=0.0'

# DDF production
if dbList_DD != '':
    cmd_ddf = '{} --dbList={}'.format(cmd_scr, dbList_DD)
    cmd_ddf += ' --outputDir={}'.format(outDir_DD)
    cmd_ddf += ' --SN_smearFlux={}'.format(smearFlux)
    cmd_ddf += ' --Fitter_sigmaz={}'.format(sigmaz)
    cmd_ddf += ' --simuParams_fromFile={}'.format(simuParams_fromFile)
    cmd_ddf += ' --simuParams_dir={}'.format(simuParams_dir_DD)
    cmd_ddf += ' --DD_list={}'.format(dd_list)
    cmd_ddf += ' --lookup_ddf={}'.format(lookup_ddf)
    cmd_ddf += ' --SN_NSNfactor={}'.format(sn_nsn_factor_dd)
    cmd_ddf += ' --Observations_coadd={}'.format(obs_coadd)
    cmd_ddf += ' --SN_sigmaz={}'.format(sn_sigmaz)
    print(cmd_ddf)
    os.system(cmd_ddf)


# WFD production
if dbList_WFD != '':
    cmd_wfd = '{} --dbList={}'.format(cmd_scr, dbList_WFD)
    cmd_wfd += ' --outputDir={}'.format(outDir_WFD)
    cmd_wfd += ' --SN_NSNfactor=10'
    cmd_wfd += ' --SN_smearFlux={}'.format(smearFlux)
    cmd_wfd += ' --Fitter_sigmaz={}'.format(sigmaz)
    cmd_wfd += ' --simuParams_fromFile={}'.format(simuParams_fromFile)
    cmd_wfd += ' --simuParams_dir={}'.format(simuParams_dir_WFD)
    cmd_wfd += ' --saturation_effect={}'.format(saturation_effect)
    cmd_wfd += ' --saturation_psf={}'.format(saturation_psf)
    cmd_wfd += ' --saturation_ccdfullwell={}'.format(saturation_ccdfullwell)
    cmd_wfd += ' --SN_z_max={}'.format(sn_z_max)
    cmd_wfd += ' --SN_NSNfactor={}'.format(sn_nsn_factor_wfd)
    cmd_wfd += ' --Observations_coadd={}'.format(obs_coadd)
    cmd_wfd += ' --SN_sigmaz={}'.format(sn_sigmaz)
    cmd_wfd += ' --fit_remove_sat={}'.format(fit_remove_sat)
    print(cmd_wfd)
    os.system(cmd_wfd)
