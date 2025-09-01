#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 09:27:44 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_utils import multiproc
from sn_tools.sn_io import checkDir
from sn_analysis.sn_tools import load_multiproc
import pandas as pd
import glob
import time


def grab_data(dataDir, dbName_DD, dbName_WFD, add_str=''):
    """
    Function to load the data using multiprocessing

    Parameters
    ----------
    dataDir : str
        Data directory.
    dbName_DD : str
        OS name for the DDF.
    dbName_WFD : str
        OS name for the WFD.
    add_str : str, optional
        To define the data dir. The default is ''.

    Returns
    -------
    df_survey : pandas df
        output data.

    """

    # load the data
    dataDir = '{}/{}_{}{}'.format(dataDir, dbName_DD, dbName_WFD, add_str)
    df_survey = pd.DataFrame()
    fis = glob.glob('{}/*.hdf5'.format(dataDir))

    params = {}
    time_ref = time.time()
    df_survey = multiproc(list(fis), params, load_multiproc, nproc=8)

    return df_survey


def process_survey(dataDir, dbName_DD, dbName_WFD, outDir):
    """
    Function to process the survey

    Parameters
    ----------
    dataDir : str
      Data directory.
    dbName_DD : str
      OS name for the DDF.
    dbName_WFD : str
      OS name for the WFD.

    Returns
    -------
    None.

    """

    df_survey = grab_data(dataDir, dbName_DD, dbName_WFD, add_str='')

    df_nospectroz = grab_data(
        dataDir, dbName_DD, dbName_WFD, add_str='_nospectroz')

    print('ll', len(df_survey), len(df_nospectroz))


parser = OptionParser(
    'script to analyze LSST SN surveys (spectro vs nospectro)')

parser.add_option('--dataDir', type=str,
                  default='../sn_surveys',
                  help='data directory [%default]')
parser.add_option('--dbList', type=str,
                  default='list_surveys.csv',
                  help='data directory [%default]')
parser.add_option('--outDir', type=str,
                  default='../sn_summary_surveys',
                  help='output directory [%default]')

opts, args = parser.parse_args()

pp = vars(opts)

# load and process the data
# load dbList

dbList = pd.read_csv(pp['dbList'], comment='#')

# loop on dblist and process

for i, row in dbList.iterrows():
    process_survey(pp['dataDir'], row['dbName_DD'],
                   row['dbName_WFD'], pp['outDir'])
