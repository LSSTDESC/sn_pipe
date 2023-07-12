#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:26:32 2023

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_analysis.sn_tools import load_complete_dbSimu
from sn_analysis.sn_calc_plot import select
from sn_tools.sn_io import checkDir
from sn_analysis.sn_selection import selection_criteria
from optparse import OptionParser
import pandas as pd


def select_filt(dataDir, dbName, sellist, seasons,
                runType='DDF_spectroz', nsn_factor=1,
                listFields='COSMOS', fieldType='DD',
                outDir='Test'):
    """
    Function to select and save selected SN data
    (per season)

    Parameters
    ----------
    dataDir : str
        location dir of data.
    dbName : str
        OS name.
    sellist : list(str)
        Selection criteria.
    seasons : list(int)
        list of seasons to process.
    runType : str, optional
        Type of run. The default is 'DDF_spectroz'.
    nsn_factor : int, optional
        MC normalization factor. The default is 1.
    listFields : list(str), optional
        list of fields to process. The default is ['COSMOS'].
    fieldType : str, optional
        Type of fields. The default is 'DD'.
    outDir : str, optional
        Main output Dir. The default is 'Test'.

    Returns
    -------
    None.

    """

    outDir_full = '{}/{}/{}'.format(outDir, dbName, runType)
    checkDir(outDir_full)

    stat_tot = pd.DataFrame()

    for seas in seasons:
        # load DDFs
        data = load_complete_dbSimu(
            dataDir, dbName, runType, listDDF=listFields, seasons=str(seas))
        print(seas, len(data))

        # apply selection on Data
        sel_data = select(data, sellist)
        sel_data = sel_data[sel_data.columns.drop(
            list(sel_data.filter(regex='mask')))]
        if 'selected' in sel_data.columns:
            sel_data = sel_data.drop(columns=['selected'])

        # save output data in pandas df
        outName = '{}/SN_{}_{}_{}.hdf5'.format(
            outDir_full, fieldType, dbName, seas)
        sel_data.to_hdf(outName, key='SN')

        # get some stat
        stat = sel_data.groupby(['field', 'season']).apply(
            lambda x: pd.DataFrame({'nsn': [len(x)/nsn_factor]})).reset_index()
        stat_tot = pd.concat((stat_tot, stat))

    stat_tot['season'] = stat_tot['season'].astype(int)
    stat_tot['nsn'] = stat_tot['nsn'].astype(int)
    outName_stat = '{}/nsn_{}.hdf5'.format(outDir_full, dbName)

    stat_tot.to_hdf(outName_stat, key='SN')


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
parser.add_option("--nsn_factor", type=int,
                  default=30, help="MC normalisation factor [%default]")

opts, args = parser.parse_args()


dataDir = opts.dataDir
dbName = opts.dbName
selconfig = opts.selconfig
runType = opts.runType
fieldType = opts.fieldType
listFields = opts.listFields
nsn_factor = opts.nsn_factor
outDir = '{}_{}'.format(dataDir, selconfig)


seasons = range(1, 11)
sellist = selection_criteria()[selconfig]

select_filt(dataDir, dbName, sellist, seasons=seasons,
            runType=runType, nsn_factor=nsn_factor,
            listFields=listFields, fieldType=fieldType, outDir=outDir)
