#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:50:30 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import glob
import pandas as pd
import os


def clean_dir(fis):

    for fi in fis:
        cmd_ = 'rm {}'.format(fi)
        os.system(cmd_)


parser = OptionParser()

parser.add_option("--dbDir", type="str", default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_test_G10_JLA/baseline_v3.0_10yrs/WFD_spectroz',
                  help="data dir [%default]")
parser.add_option("--timescale", type=str,
                  default='year',
                  help="Time scale for NSN estimation. [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
timescale = opts.timescale


for season in range(1, 15, 1):
    fis = glob.glob('{}/*{}_{}.hdf5'.format(dbDir, timescale, season))
    print(season, len(fis))
    if len(fis) > 0:
        r = []
        for bb in fis:
            r.append(bb)
        common_substring = os.path.commonprefix(
            [fis[0], fis[1], fis[2], fis[4]])

        outName = '_'.join(common_substring.split('/')[-1].split('_')[:-1])
        print(common_substring)
        outName = '{}/{}_{}_{}.hdf5'.format(dbDir, outName, timescale, season)
        df = pd.DataFrame()
        for fi in fis:
            dd = pd.read_hdf(fi)
            df = pd.concat((df, dd))
        df.to_hdf(outName, key='sn')

    clean_dir(fis)
