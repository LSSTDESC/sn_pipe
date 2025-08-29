#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 15:52:55 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import pandas as pd
import time
from sn_tools.sn_utils import multiproc
from sn_tools.sn_io import checkDir


def load_multiproc(fis, params, j=0, output_q=None):
    """
    Function to load data using multiprocessing

    Parameters
    ----------
    fis : list(str)
        list of files to load.
    params : dict
        parameters.
    j : int, optional
        internal tag for multiprocessing. The default is 0.
    output_q : output processing queue, optional
        where to store the data. The default is None.

    Returns
    -------
    pandas df
        output data.

    """

    df_survey = pd.DataFrame()
    for fi in fis:
        dd_ = pd.read_hdf(fi)
        df_survey = pd.concat((df_survey, dd_))

    if output_q is not None:
        return output_q.put({j: df_survey})
    else:
        return df_survey


def get_stat(grp):
    """
    Estimate some stat

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    pandas df
        result.

    """

    dd = {}
    dd['nsn'] = [len(grp)]

    survey_area = grp['survey_area'].mean()
    survey_area *= len(grp['healpixID'].unique())
    dd['survey_area'] = [survey_area]

    idx = grp['z_fit'] >= 0.8
    sela = grp[idx]
    dd['nsn_z_08'] = [len(sela)]
    idx &= grp['sigmaC'] <= 0.04

    sel = grp[idx]

    dd['nsn_z_08_sigmaC'] = [len(sel)]

    return pd.DataFrame.from_dict(dd)


def get_statb(grp):
    """
    Estimate some stat on SN realization

    Parameters
    ----------
    grp : pandas df
        Data to process.

    Returns
    -------
    pandas df
        output data.

    """

    dd = {}
    dd['nsn'] = [grp['nsn'].mean()]
    dd['err_nsn'] = [grp['nsn'].std()]
    dd['nsn_z_08_sigmaC'] = [grp['nsn_z_08_sigmaC'].mean()]
    dd['err_nsn_z_08_sigmaC'] = [grp['nsn_z_08_sigmaC'].std()]
    dd['nsn_z_08'] = [grp['nsn_z_08'].mean()]
    dd['err_nsn_z_08'] = [grp['nsn_z_08'].std()]
    dd['survey_area'] = [grp['survey_area'].mean()]

    return pd.DataFrame.from_dict(dd)


def process_survey(dataDir, dbName_DD, dbName_WFD, outDir):
    """
    Main function to process the data

    Parameters
    ----------
    dataDir : str
        Data directory.
    dbName_DD : str
        OS name for the DD survey.
    dbName_WFD : str
        Os name for the WFD survey.
    outDir : str
        Main output directory.

    Returns
    -------
    None.

    """

    # load the data
    dataDir = '{}/{}_{}'.format(dataDir, dbName_DD, dbName_WFD)
    df_survey = pd.DataFrame()
    fis = glob.glob('{}/*.hdf5'.format(dataDir))

    params = {}
    time_ref = time.time()
    df_survey = multiproc(list(fis), params, load_multiproc, nproc=8)

    print('data loaded', dbName_DD, time.time()-time_ref)

    # estimate some stat

    dfb = df_survey.groupby(['field', 'year', 'nreal']).apply(
        lambda x: get_stat(x)).reset_index()

    dfc = dfb.groupby(['field', 'year']).apply(
        lambda x: get_statb(x)).reset_index()
    dfc['dbName'] = dbName_DD

    outDir = '{}/{}'.format(outDir, dbName_DD)
    checkDir(outDir)

    outName = '{}/sn_survey.hdf5'.format(outDir)

    dfc.to_hdf(outName, key='sn_survey')


parser = OptionParser('script to analyze LSST SN surveys')

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

# load dbList

dbList = pd.read_csv(pp['dbList'], comment='#')

# loop on dblist and process

for i, row in dbList.iterrows():
    process_survey(pp['dataDir'], row['dbName_DD'],
                   row['dbName_WFD'], pp['outDir'])
