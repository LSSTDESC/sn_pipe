#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:13:13 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import glob
from sn_tools.sn_utils import multiproc


def calc(dbNames, params, j, output_q):
    """
    Function to process dbNames

    Parameters
    ----------
    dbNames : list(str)
        List of dbNames to process.
    params : dict
        Parameters for the process.
    j : int
        internal tag for multipticessing.
    output_q : multiprocessing queue
        Queue for multiprocessing.

    Returns
    -------
    pandas df
        Output data.

    """

    dbDir = params['dbDir']
    runType = params['runType']
    df = pd.DataFrame()
    for dbName in dbNames:
        dfb = calc_indiv(dbDir, dbName, runType)
        df = pd.concat((df, dfb))

    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df


def calc_indiv(dbDir, dbName, runType):
    """
    Function to estimate the number of SN for a dbName

    Parameters
    ----------
    dbDir : str
        Data dir .
    dbName : str
        dbName to process.
    runType : str
        Type of run.

    Returns
    -------
    res : pandas df
        output data.

    """

    full_path = '{}/{}/{}/*.hdf5'.format(dbDir, dbName, runType)

    fis = glob.glob(full_path)

    if len(fis) == 0:
        print('Problem: file not found for', full_path)

    assert len(fis) > 0

    nsn = 0
    for fi in fis:
        tt = pd.read_hdf(fi)
        nsn += len(tt)

    res = pd.DataFrame([nsn], columns=['nsn'])
    res['dbName'] = dbName

    return res


parser = OptionParser(
    description='Script to estimate the number of simulated SNe Ia')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='list_OS.csv',
                  help='list of dbs to process [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='run type  [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='nproc for multiprocessing  [%default]')
parser.add_option('--outName', type=str,
                  default='nsn_WFD.csv',
                  help='output name for the results  [%default]')

opts, args = parser.parse_args()


dbDir = opts.dbDir
dbList = opts.dbList
runType = opts.runType
nproc = opts.nproc
outName = opts.outName

data = pd.read_csv(dbList, comment='#')
params = {}

params['dbDir'] = dbDir
params['runType'] = runType


res = multiproc(data['dbName'].to_list(), params, calc, nproc)

print(res)

res.to_csv(outName, index=False)
