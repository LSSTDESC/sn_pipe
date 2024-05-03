#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  11 09:36:32 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_analysis.sn_tools import load_complete_dbSimu
from sn_tools.sn_io import checkDir
from optparse import OptionParser
import glob
import pandas as pd


def get_simu_params_DDF(dataDir, dbName, seasons,
                        runType='DDF_spectroz',
                        listFields='COSMOS', fieldType='DD',
                        outDir='Test', nproc=8,
                        dataType='pandasDataFrame', suffix=''):
    """

    Function to extract simulation parameters

    Parameters
    ----------
    dataDir : str
        Location dir of the data.
    dbName : str
        OS to process.
    seasons : list(int)
        List of seasons to process
    runType : str, optional
        Runtype. The default is 'DDF_spectroz'.
    listFields : list(str), optional
        List of fields to process. The default is 'COSMOS'.
    fieldType : str, optional
        Field type to process. The default is 'DD'.
    outDir : str, optional
        output directory. The default is 'Test'.
    nproc : int, optional
        number of procs to process. The default is 8.
    dataType : str, optional
        Type of data. The default is 'pandasDataFrame'.
    suffix : str, optional
        Suffix for data to process. The default is ''.

    Returns
    -------
    None.

    """

    outDir_full = '{}/{}/{}'.format(outDir, dbName, runType)
    checkDir(outDir_full)

    print('output directory', outDir_full)

    timescale = 'season'
    ccols = ['healpixID', 'z', 'x1', 'color', 'daymax', 'x0', 'season',
             'epsilon_x0', 'epsilon_x1', 'epsilon_color', 'epsilon_daymax',
             'SNID']

    for seas in seasons:
        # load DDFs

        data = load_complete_dbSimu(
            dataDir, dbName, runType, listDDF=listFields,
            seasons=str(seas), nproc=nproc, dataType=dataType, suffix=suffix)
        print(seas, len(data))

        if data.empty:
            continue

        # extract simulation parameters from data
        outName = '{}/SN_simu_params_{}_{}_{}_{}{}.hdf5'.format(
            outDir_full, fieldType, dbName, timescale, seas, suffix)

        data[ccols].to_hdf(outName, key='simu', index=False)


def get_simu_params_WFD(dataDir, dbName, seasons,
                        runType='DDF_spectroz',
                        listFields='COSMOS', fieldType='DD',
                        outDir='Test', nproc=8,
                        dataType='pandasDataFrame', suffix=''):
    """

    Function to extract simulation parameters

    Parameters
    ----------
    dataDir : str
        Location dir of the data.
    dbName : str
        OS to process.
    seasons : list(int)
        List of seasons to process
    runType : str, optional
        Runtype. The default is 'DDF_spectroz'.
    listFields : list(str), optional
        List of fields to process. The default is 'COSMOS'.
    fieldType : str, optional
        Field type to process. The default is 'DD'.
    outDir : str, optional
        output directory. The default is 'Test'.
    nproc : int, optional
        number of procs to process. The default is 8.
    dataType : str, optional
        Type of data. The default is 'pandasDataFrame'.
    suffix : str, optional
        Suffix for data to process. The default is ''.

    Returns
    -------
    None.

    """

    outDir_full = '{}/{}/{}'.format(outDir, dbName, runType)
    checkDir(outDir_full)

    print('output directory', outDir_full)

    timescale = 'season'
    ccols = ['healpixID', 'z', 'x1', 'color', 'daymax', 'x0', 'season',
             'epsilon_x0', 'epsilon_x1', 'epsilon_color', 'epsilon_daymax',
             'SNID']

    ddataDir = '{}/{}/{}'.format(dataDir, dbName, runType)
    fis = glob.glob('{}/*.hdf5'.format(ddataDir))

    ddict = {}
    for seas in seasons:
        ddict[seas] = pd.DataFrame()

    print('ooo', fis, outDir_full)
    for fi in fis:
        df = pd.read_hdf(fi)

        for seas in seasons:
            idx = df['season'] == seas
            dfs = df[idx]
            ddict[seas] = pd.concat((ddict[seas], dfs))

    data_tot = pd.DataFrame()
    for seas, data in ddict.items():

        outName = '{}/SN_simu_params_{}_{}_{}_{}{}.hdf5'.format(
            outDir_full, fieldType, dbName, timescale, seas, suffix)

        data[ccols].to_hdf(outName, key='simu', index=False)
        data_tot = pd.concat((data_tot, data[ccols]))

    outName = '{}/SN_simu_params_{}_{}_{}{}.hdf5'.format(
        outDir_full, fieldType, dbName, timescale, suffix)

    data_tot.to_hdf(outName, key='simu', index=False)

    """
    for seas in seasons:
        # load DDFs

        data = load_complete_dbSimu(
            dataDir, dbName, runType, listDDF=listFields,
            seasons=str(seas), nproc=nproc, dataType=dataType, suffix=suffix)
        print(seas, len(data))

        if data.empty:
            continue

        # extract simulation parameters from data
        outName = '{}/SN_simu_params_{}_{}_{}_{}{}.hdf5'.format(
            outDir_full, fieldType, dbName, timescale, seas, suffix)

        data[ccols].to_hdf(outName, key='simu', index=False)
    """


parser = OptionParser()

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell',
                  help="data dir[%default]")
parser.add_option("--dbName", type=str,
                  default='DDF_Univ_WZ', help="db name [%default]")
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help=" [%default]")
parser.add_option("--runType", type=str,
                  default='DDF_spectroz', help=" [%default]")
parser.add_option("--listFields", type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help=" [%default]")
parser.add_option("--fieldType", type=str,
                  default='DDF',
                  help=" [%default]")
parser.add_option("--nproc", type=int,
                  default=8, help="nproc for multiprocessing [%default]")
parser.add_option("--dataType", type=str,
                  default='pandasDataFrame',
                  help="Data type to process (pandasDataFrame/astropyTable). [%default]")
parser.add_option("--suffix", type=str,
                  default='',
                  help="suffix of the data file name to process. [%default]")

opts, args = parser.parse_args()

dataDir = opts.dataDir
dbName = opts.dbName
selconfig = opts.selconfig
runType = opts.runType
fieldType = opts.fieldType
listFields = opts.listFields
dataType = opts.dataType
suffix = opts.suffix

outDir = '{}_simuparams'.format(dataDir)
nproc = opts.nproc
seasons = range(1, 13)

if fieldType == 'WFD':
    get_simu_params_WFD(dataDir, dbName, seasons=seasons,
                        runType=runType,
                        listFields=listFields, fieldType=fieldType,
                        outDir=outDir, nproc=nproc,
                        dataType=dataType, suffix=suffix)
else:
    get_simu_params_DDF(dataDir, dbName, seasons=seasons,
                        runType=runType,
                        listFields=listFields, fieldType=fieldType,
                        outDir=outDir, nproc=nproc,
                        dataType=dataType, suffix=suffix)
