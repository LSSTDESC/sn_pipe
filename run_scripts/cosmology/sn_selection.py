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
import numpy as np


def select_filt(dataDir, dbName, sellist, seasons,
                runType='DDF_spectroz', nsn_factor=1,
                listFields='COSMOS', fieldType='DD',
                outDir='Test', nproc=8, mjdStart=60796.001,
                timescale='year'):
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
    nproc: int, optional.
      number of procs for multiprocessing. The default is 8.
    mjdStart: float, optional.
      starting date of the LSST survey. The default is 60796.001.
    timescale : str, optional
        timescale for calculation. The default is 'year'.

    Returns
    -------
    None.

    """

    outDir_full = '{}/{}/{}'.format(outDir, dbName, runType)
    checkDir(outDir_full)

    # remove files if necessary
    import os
    store = {}
    for seas in seasons:
        outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
            outDir_full, fieldType, dbName, timescale, seas)
        """
        if os.path.isfile(outName):
            os.remove(outName)
        """
        store[seas] = pd.HDFStore(outName, 'w')

    stat_tot = pd.DataFrame()
    # sel_tot = pd.DataFrame()

    for seas in seasons:
        # load DDFs
        data = load_complete_dbSimu(
            dataDir, dbName, runType, listDDF=listFields,
            seasons=str(seas), nproc=nproc)
        print(seas, len(data))

        if data.empty:
            continue
        # apply selection on Data
        sel_data = select(data, sellist)
        sel_data = sel_data[sel_data.columns.drop(
            list(sel_data.filter(regex='mask')))]
        if 'selected' in sel_data.columns:
            sel_data = sel_data.drop(columns=['selected'])

        # add lsststart as column
        sel_data['LSSTStart'] = mjdStart
        # add year as column
        sel_data['year'] = sel_data['daymax']+60*(1.+sel_data['z'])
        sel_data['year'] -= mjdStart
        sel_data['year'] /= 365.
        sel_data['year'] += 1.
        sel_data['year'] = sel_data['year'].astype(int)

        # sel_tot = pd.concat((sel_data, sel_tot))

        # save output data in pandas df
        if timescale == 'season':
            store[seas].put('SN', sel_data)
            """
            outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
                outDir_full, fieldType, dbName, timescale, seas)

            sel_data.to_hdf(outName, key='SN')
            """
        else:
            years = sel_data[timescale].unique()
            for vv in years:
                idx = sel_data[timescale] == vv
                selb = sel_data[idx]
                store[vv].put('SN', selb)

        """
        # sel_tot = pd.concat((sel_data, sel_tot))
        # get some stat
        # stat = sel_data.groupby(['field', 'season']).apply(
        #     lambda x: pd.DataFrame({'nsn': [len(x)/nsn_factor]})).reset_index()
        """
        stat = get_stat(sel_data, nsn_factor, timescale=timescale)
        stat_tot = pd.concat((stat_tot, stat))

    if timescale == 'year':
        stat_tot = stat_tot.groupby(['field', timescale])[
            'nsn', 'nsn_z_0.1', 'nsn_z_0.2'].sum().reset_index()
    """
    if sel_tot.empty:
        return -1

    timescales = sel_tot[timescale].unique()

    for timescalev in timescales:
        idx = sel_tot[timescale] == timescalev
        sel_data = sel_tot[idx]
        # save output data in pandas df
        outName = '{}/SN_{}_{}_{}.hdf5'.format(
            outDir_full, fieldType, dbName, timescalev)

        sel_data.to_hdf(outName, key='SN')
        stat = get_stat(sel_data, nsn_factor, timescale=timescale)
        stat_tot = pd.concat((stat_tot, stat))

    # stat_tot = get_stat(sel_tot, nsn_factor, timescale=timescale)
    """

    stat_tot[timescale] = stat_tot[timescale].astype(int)
    stat_tot['nsn'] = stat_tot['nsn'].astype(int)
    outName_stat = '{}/nsn_{}_{}.hdf5'.format(outDir_full, dbName, timescale)
    store = pd.HDFStore(outName_stat, 'w')
    store.put('SN', stat_tot)

    #stat_tot.to_hdf(outName_stat, key='SN')


def get_stat(sel_data, nsn_factor, timescale='year'):
    """
    Function to estimate nsn

    Parameters
    ----------
    sel_data : pandas df
        Data to process.
    nsn_factor : float
        Normalization factor.
    timescale: str, optional.
      time scale for nsn estimation. The default is 'year'.

    Returns
    -------
    # stat_sn : pandas df
        nsn results .

    """

    # get total nsn
    stat_sn = sel_data.groupby(['field', timescale]).apply(
        lambda x: nsn_estimate(x,
                               zmax=1.1,
                               nsn_factor=nsn_factor,
                               varname='nsn')).reset_index()
    stat_sn = stat_sn.drop(['level_2'], axis=1)

    for zlim in [0.1, 0.2]:
        nname = 'nsn_z_{}'.format(np.round(zlim, 1))
        stat_sn_z = sel_data.groupby(['field', timescale]).apply(
            lambda x: nsn_estimate(x,
                                   zmax=zlim,
                                   nsn_factor=nsn_factor,
                                   varname=nname)).reset_index()
        stat_sn_z = stat_sn_z.drop(['level_2'], axis=1)
        # merge
        stat_sn = stat_sn.merge(
            stat_sn_z, left_on=['field', timescale],
            right_on=['field', timescale], suffixes=['', ''])

    return stat_sn


def nsn_estimate(grp, zmax=1.1, nsn_factor=1, varname='nsn'):

    idx = grp['z'] <= zmax

    sel = grp[idx]

    res = np.rint(len(sel)/nsn_factor)

    return pd.DataFrame({varname: [res]})


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
parser.add_option("--nproc", type=int,
                  default=8, help="nproc for multiprocessing [%default]")
parser.add_option("--lsst_start", type=str,
                  default='LSSTStart.csv',
                  help="List of LSST start survey night. [%default]")
parser.add_option("--timescale", type=str,
                  default='year',
                  help="Time scale for NSN estimation. [%default]")

opts, args = parser.parse_args()


dataDir = opts.dataDir
dbName = opts.dbName
selconfig = opts.selconfig
runType = opts.runType
fieldType = opts.fieldType
listFields = opts.listFields
nsn_factor = opts.nsn_factor
lsst_start = opts.lsst_start
timescale = opts.timescale

outDir = '{}_{}'.format(dataDir, selconfig)
nproc = opts.nproc


seasons = range(1, 12)

sellist = selection_criteria()[selconfig]

mjd_start = pd.read_csv(lsst_start, comment='#')

idx = mjd_start['dbName'] == dbName
mjdStart = mjd_start[idx]['LSSTStart'].median()

select_filt(dataDir, dbName, sellist, seasons=seasons,
            runType=runType, nsn_factor=nsn_factor,
            listFields=listFields, fieldType=fieldType,
            outDir=outDir, nproc=nproc, mjdStart=mjdStart,
            timescale=timescale)
