#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:26:32 2023

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_analysis.sn_selection import Select_filt
from sn_analysis.sn_selection import selection_criteria
from optparse import OptionParser
import pandas as pd
import numpy as np
import glob
import os


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
    if 'level_2' in stat_sn.columns:
        stat_sn = stat_sn.drop(['level_2'], axis=1)

    # for zlim in [0.1, 0.2]:
    zlim = np.arange(0.0, 1.1, 0.1)
    r = []
    for i in range(len(zlim)-1):
        zmin = zlim[i]
        zmax = zlim[i+1]
        nname = 'nsn_z_{}_{}'.format(np.round(zmin, 1), np.round(zmax, 1))
        r.append(nname)
        stat_sn_z = sel_data.groupby(['field', timescale]).apply(
            lambda x: nsn_estimate(x,
                                   zmin=zmin,
                                   zmax=zmax,
                                   nsn_factor=nsn_factor,
                                   varname=nname)).reset_index()
        if 'level_2' in stat_sn_z.columns:
            stat_sn_z = stat_sn_z.drop(['level_2'], axis=1)
        # merge
        stat_sn = stat_sn.merge(
            stat_sn_z, left_on=['field', timescale],
            right_on=['field', timescale], suffixes=['', ''])

    return stat_sn, r


def nsn_estimate(grp, zmin=0., zmax=1.1, nsn_factor=1, varname='nsn'):
    """
    Method to estimate the number of sn in a redshift range

    Parameters
    ----------
    grp : pandas df
        Data to process.
    zmin : float, optional
        Min redshift. The default is 0..
    zmax : float optional
        Max redshift. The default is 1.1.
    nsn_factor : int, optional
        Normalization parameter. The default is 1.
    varname : str, optional
        Column of interest. The default is 'nsn'.

    Returns
    -------
    pandas df
        Two columns: column of interest, nsn.

    """

    idx = grp['z'] < zmax
    idx &= grp['z'] >= zmin

    sel = grp[idx]

    res = np.rint(len(sel)/nsn_factor)

    return pd.DataFrame({varname: [res]})


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

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell',
                  help="data dir[%default]")
parser.add_option("--outDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help="output dir[%default]")
parser.add_option("--dbName", type=str,
                  default='DDF_Univ_WZ', help="db name [%default]")
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help="sel config name[%default]")
parser.add_option("--zType", type=str,
                  default='spectroz', help="z type (spectroz/photz) [%default]")
parser.add_option("--listFields", type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help=" [%default]")
parser.add_option("--fieldType", type=str,
                  default='DDF',
                  help=" [%default]")
"""
parser.add_option("--nsn_factor", type=int,
                  default=30, help="MC normalisation factor [%default]")
"""
parser.add_option("--nproc", type=int,
                  default=8, help="nproc for multiprocessing [%default]")
parser.add_option("--timescale", type=str,
                  default='year',
                  help="Time scale for NSN estimation. [%default]")
parser.add_option("--dataType", type=str,
                  default='pandasDataFrame',
                  help="Data type to process (pandasDataFrame/astropyTable). [%default]")
parser.add_option("--ebvofMW", type=float,
                  default=0.25,
                  help="Max e(B-V). [%default]")
parser.add_option("--seasons", type=str,
                  default='1_14',
                  help="seasons to process [%default]")

opts, args = parser.parse_args()


dataDir = opts.dataDir
dbName = opts.dbName
selconfig = opts.selconfig
zType = opts.zType
fieldType = opts.fieldType
listFields = opts.listFields
# nsn_factor = opts.nsn_factor
timescale = opts.timescale
dataType = opts.dataType
ebvofMW = opts.ebvofMW
outDir = opts.outDir
seasons = opts.seasons.split(',')

nproc = opts.nproc


# seasons = range(1, 13)

sellist = selection_criteria()[selconfig]

"""
select_filt(dataDir, dbName, sellist, seasons=seasons,
            zType=zType, nsn_factor=nsn_factor,
            listFields=listFields, fieldType=fieldType,
            outDir=outDir, nproc=nproc,
            timescale=timescale,
            dataType=dataType,
            ebvofMW=ebvofMW)
"""
Select_filt(dataDir, dbName, sellist, seasons=seasons,
            zType=zType,  # nsn_factor=nsn_factor,
            listFields=listFields, fieldType=fieldType,
            outDir=outDir, nproc=nproc,
            timescale=timescale,
            dataType=dataType,
            ebvofMW=ebvofMW)
if fieldType == 'WFD':
    # merge and clean
    print('merging and cleaning', outDir)
    dataDir = '{}/{}/{}_{}'.format(outDir, dbName, fieldType, zType)
    print('merging and cleaning', dataDir)
    process(dataDir, timescale)
