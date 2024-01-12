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
parser.add_option("--simuParams_fromFile", type=int, default=0,
                  help="to use simulation params from file [%default]")
parser.add_option("--simuParams_dir", type=str,
                  default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_nnew',
                  help="Dir for the simu param files [%default]")

opts, args = parser.parse_args()

dbList_DD = opts.dbList_DD
outDir_DD = opts.outDir_DD

dbList_WFD = opts.dbList_WFD
outDir_WFD = opts.outDir_WFD

smearFlux = opts.SN_smearFlux
sigmaz = opts.Fitter_sigmaz
simuParams_fromFile = opts.simuParams_fromFile
simuParams_dir = opts.simuParams_dir

cmd_scr = 'python for_batch/scripts/sn_prod/loop_prod.py'
cmd_scr += ' --SN_sigmaInt=0.0'

# DDF production
if dbList_DD != '':
    cmd_ddf = '{} --dbList={}'.format(cmd_scr, dbList_DD)
    cmd_ddf += ' --outputDir={}'.format(outDir_DD)
    cmd_ddf += ' --SN_smearFlux={}'.format(smearFlux)
    cmd_ddf += ' --Fitter_sigmaz={}'.format(sigmaz)
    cmd_ddf += ' --simuParams_fromFile={}'.format(simuParams_fromFile)
    cmd_ddf += ' --simuParams_dir={}'.format(simuParams_dir)

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
    cmd_wfd += ' --simuParams_dir={}'.format(simuParams_dir)

    print(cmd_wfd)
    os.system(cmd_wfd)
