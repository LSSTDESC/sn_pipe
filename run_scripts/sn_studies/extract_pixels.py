#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:27:21 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import pandas as pd
import glob
from sn_tools.sn_io import checkDir
from sn_tools.sn_utils import multiproc


def process_file(fis, params, j=0, output_q=None):
    """
    Function to process files

    Parameters
    ----------
    fis : list(str)
        List of files to process (full path).
    params : dict
        parameters.
    j : int, optional
        internal tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        where to put the results. The default is None.

    Returns
    -------
    dict/pandas df
        results.

    """

    cols = params['cols']
    drop_duplicates = params['drop_duplicates']

    df_res = pd.DataFrame()
    for fi in fis:
        df = pd.read_hdf(fi)
        df = df[cols]
        if drop_duplicates:
            df = df.drop_duplicates()
        df_res = pd.concat((df_res, df))

    if output_q is not None:
        return output_q.put({j: df_res})
    else:
        return df_res


parser = OptionParser(description='Script to extract pixel infos')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA',
                  help='OS location dir [%default]')
parser.add_option('--dbList', type=str,
                  default='config_ana_selplot.csv',
                  help='list of OS to process [%default]')
parser.add_option('--fieldType', type=str,
                  default='DDF',
                  help='type of field to process [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='type of run [%default]')
parser.add_option('--outDir', type=str,
                  default='../pixels_DDF',
                  help='output Dir [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='nproc for multiprocessing [%default]')
parser.add_option('--drop_duplicate', type=int,
                  default=0,
                  help='to drop duplicate [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
fieldType = opts.fieldType
runType = opts.runType
outDir = opts.outDir
nproc = opts.nproc
drop_duplicate = opts.drop_duplicate

# create outDir if necessary
checkDir(outDir)

# load config
df_config = pd.read_csv(dbList, comment='#')

cols = ['healpixID', 'pixRA', 'pixDec', 'season', 'season_length']
cols += ['survey_area', 'x1', 'color']

params = {}
params['cols'] = cols
params['drop_duplicates'] = drop_duplicate

for i, row in df_config.iterrows():
    dbName = row['dbName']
    dirName = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, runType)
    df_res = pd.DataFrame()
    fis = glob.glob('{}/*.hdf5'.format(dirName))
    df_res = multiproc(fis, params, process_file, nproc)

    """
    for fi in fis:
        df = pd.read_hdf(fi)
        df = df[cols]
        # df = df.drop_duplicates()
        df_res = pd.concat((df_res, df))
    # df_res = df_res.drop_duplicates()
    """
    if drop_duplicate:
        df_res = df_res.drop_duplicates()
    print(df_res)
    outName = '{}/{}.hdf5'.format(outDir, dbName)
    df_res.to_hdf(outName, key='pixels')
