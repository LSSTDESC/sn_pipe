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


def process(dataDir, timescale):
    """
    Function to process data

    Parameters
    ----------
    dbDir : str
        Data dir.
    timescale : str
        Timescale (year/season).

    Returns
    -------
    None.

    """
    for season in range(1, 15, 1):
        fis = glob.glob('{}/*{}_{}.hdf5'.format(dataDir, timescale, season))
        print(season, len(fis))
        if len(fis) > 0:
            r = []
            for bb in fis:
                r.append(bb)
            common_substring = os.path.commonprefix(
                [fis[0], fis[1], fis[2], fis[4]])

            outName = '_'.join(common_substring.split('/')[-1].split('_')[:-1])
            outName = '{}/{}_{}_{}.hdf5'.format(dataDir,
                                                outName, timescale, season)
            df = pd.DataFrame()
            for fi in fis:
                dd = pd.read_hdf(fi)
                df = pd.concat((df, dd))
            df.to_hdf(outName, key='sn')

    clean_dir(fis)


def clean_dir(fis):
    """
    Function to remove files

    Parameters
    ----------
    fis : list(str)
        List of files to remove.

    Returns
    -------
    None.

    """

    for fi in fis:
        cmd_ = 'rm {}'.format(fi)
        os.system(cmd_)


parser = OptionParser()

parser.add_option("--dbDir", type="str",
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_test_G10_JLA',
                  help="data dir [%default]")
parser.add_option("--dbList", type=str,
                  default='list_OS.csv', help="DB list to process [%default]")
parser.add_option("--timescale", type=str,
                  default='year',
                  help="Time scale for NSN estimation. [%default]")
parser.add_option("--zType", type=str,
                  default='spectroz', help="z type (spectroz/photz)  [%default]")
parser.add_option("--fieldType", type=str,
                  default='WFD', help="field type (DDF/WFD)  [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
timescale = opts.timescale
zType = opts.zType
fieldType = opts.fieldType


# load dbs
dbNames = pd.read_csv(dbList, comment='#')

# loop on dbs and process

for i,row in dbNames.iterrows():
    dbName = row['dbName']
    dataDir = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, zType)
    process(dataDir, timescale)
