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
import numpy as np
import os
from sn_tools.sn_utils import multiproc
from sn_tools.sn_io import checkDir
from sn_analysis.sn_tools import load_multiproc, get_stat, get_statb
from sn_analysis.sn_calc_plot import bin_it


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

    # data of the survey - with spectro scenario
    df_survey = grab_data(dataDir, dbName_DD, dbName_WFD)
    statIt(df_survey, dbName_DD, outDir)

    # check whether all survey data are available
    add_str = '_nospectroz'
    outDir_all = '{}/{}_{}{}'.format(dataDir, dbName_DD, dbName_WFD, add_str)
    if os.path.isdir(outDir_all):
        df_all = grab_data(dataDir, dbName_DD, dbName_WFD, add_str)
        statIt(df_all, dbName_DD, outDir, fName='sn_survey_all.hdf5')
        ana_fields(df_survey, df_all, outDir, dbName_DD)


def grab_data(dataDir, dbName_DD, dbName_WFD, add_str=''):
    """
    To grab the data

    Parameters
    ----------
    dataDir : str
        Data dir.
    dbName_DD : str
        OS for the DDF survey.
    dbName_WFD : str
        OS for the WFD survey.
    add_str : str, optional
        addendum to fName. The default is ''.

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

    df_survey = multiproc(list(fis), params, load_multiproc, nproc=8)

    if 'sigmaC' not in df_survey.columns:
        df_survey['sigmaC'] = np.sqrt(df_survey['Cov_colorcolor'])

    return df_survey


def statIt(df_survey, dbName_DD, outDir, fName='sn_survey.hdf5'):
    """
    Estimate and save statistics on file

    Parameters
    ----------
    df_survey : pandas df
        Data to process
    dbName_DD : str
        OS for the DDF survey.
    outDir : str
        Output directory.
    fName : str, optional
        Output file name. The default is 'sn_survey.hdf5'.

    Returns
    -------
    None.

    """

    # estimate some stat

    dfb = df_survey.groupby(['field', 'year', 'nreal']).apply(
        lambda x: get_stat(x)).reset_index()

    dfc = dfb.groupby(['field', 'year']).apply(
        lambda x: get_statb(x)).reset_index()
    dfc['dbName'] = dbName_DD

    outDir = '{}/{}'.format(outDir, dbName_DD)
    checkDir(outDir)

    outName = '{}/{}'.format(outDir, fName)

    dfc.to_hdf(outName, key='sn_survey')


def ana_fields(df_survey, df_nospectroz, outDir,
               dbName, fName='effi_spectro.hdf5'):
    """
    Method to analyze the field (spectro efficiency)

    Parameters
    ----------
    df_survey : pandas df
        Data to process.
    df_nospectroz : pandas df
        Data to process.
    outDir : str
        output directory.
    dbName : str
        OS name.
    fName : str, optional
        output file name. The default is 'effi_spectro.hdf5'.

    Returns
    -------
    None.

    """

    fields = df_nospectroz['field'].unique()

    res = pd.DataFrame()

    for field in fields:
        dfa = df_survey[df_survey['field'] == field]
        dfb = df_nospectroz[df_nospectroz['field'] == field]
        years = dfb['year'].unique()
        for year in years:
            dfaa = dfa[dfa['year'] == year]
            dfbb = dfb[dfb['year'] == year]
            rr = get_effi(dfaa, dfbb, field=field)
            rr['year'] = year
            res = pd.concat((res, rr))

    res['dbName'] = dbName
    outName = '{}/{}/{}'.format(outDir, dbName, fName)

    res.to_hdf(outName, key='effi')


def get_effi(df_survey, df_nospectroz, field='COSMOS'):
    """
    Estimate spectroscopic efficiencies

    Parameters
    ----------
    df_survey : pandas df
        Data to process.
    df_nospectroz : pandas df
        Data to process.
    field : str, optional
        Field name. The default is 'COSMOS'.

    Returns
    -------
    pandas df
        Results (efficiency+error).

    """

    idxa = df_survey['field'] == field
    dfa = df_survey[idxa]

    idxb = df_nospectroz['field'] == field
    dfb = df_nospectroz[idxb]

    bins = np.arange(0.01, 1.15, 0.05)

    resa = bin_it(dfa, xvar='z_fit', bins=bins)

    resb = bin_it(dfb, xvar='z_fit', bins=bins)

    resc = resa.merge(resb, left_on=['z_fit'], right_on=['z_fit'])

    resc['effi'] = resc['NSN_x']/resc['NSN_y']
    var_effi = resc['NSN_y']*resc['effi']*(1.-resc['effi'])/resc['NSN_y']**2
    resc['err_effi'] = np.sqrt(var_effi)
    resc['field'] = field

    outcols = ['field', 'effi', 'err_effi', 'z_fit']
    return resc[outcols]


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
