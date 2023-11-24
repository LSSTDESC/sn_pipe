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
parser.add_option("--outDir_WFD", type="str", default='/sps/lsst/users/gris/Output_SN_WFD_sigmaInt_0.0_Hounsell_new',
                  help="output dir for WFD prod [%default]")
opts, args = parser.parse_args()

dbList_DD = opts.dbList_DD
outDir_DD = opts.outDir_DD

dbList_WFD = opts.dbList_WFD
outDir_WFD = opts.outDir_WFD

cmd_scr = 'python for_batch/scripts/sn_prod/loop_prod.py'
cmd_scr += ' --SN_sigmaInt=0.0'

# DDF production
if dbList_DD != '':
    cmd_ddf = '{} --dbList={}'.format(cmd_scr, dbList_DD)
    cmd_ddf += ' --outputDir={}'.format(outDir_DD)

    print(cmd_ddf)
    os.system(cmd_ddf)


# WFD production
if dbList_WFD != '':
    cmd_wfd = '{} --dbList={}'.format(cmd_scr, dbList_WFD)
    cmd_wfd += ' --outputDir={}'.format(outDir_WFD)
    cmd_wfd += ' --SN_NSNfactor=10'

    print(cmd_wfd)
    os.system(cmd_wfd)
