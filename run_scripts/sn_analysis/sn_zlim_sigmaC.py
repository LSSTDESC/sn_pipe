#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:34:47 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from optparse import OptionParser
import glob
import numpy as np
from sn_analysis.sn_calc_plot import bin_it_mean
from scipy.interpolate import interp1d
from sn_tools.sn_utils import multiproc
from sn_tools.sn_io import checkDir


def load_data(dbDir, dbName, runType, timescale, season):
    """
    Function to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        dbname.
    runType : str
        run type.
    timescale : str
        time scale.
    season: int
        season number.

    Returns
    -------
    df : pandas df
        loaded data.

    """

    mainDir = '{}/{}/{}'.format(dbDir, dbName, runType)

    df = pd.DataFrame()

    print('loading', dbName, 'season', season)
    fis = glob.glob('{}/*_season_{}.hdf5'.format(mainDir, season))

    for fi in fis:
        the_df = pd.read_hdf(fi)
        df = pd.concat((df, the_df))

    return df


def process(dbDir, dbName, runType, timescale, outDir, sigmaC=0.04, nproc=8,
            seasons=range(1, 11)):
    """
    Function to process the data

    Parameters
    ----------
    dbDir : str
        data dir.
    dbName : str
        OS name.
    runType : str
        run type.
    timescale : str
        timescale.
    outDir: str
        output directory
    sigmaC : float, optional
           sigma color selection value. The default is 0.04.
    nproc: int, optional.
        number of procs for multiprocessing.
    seasons: list(int)
        list of seasons to process. The default is range(1,11)

    Returns
    -------
    tt : pandas df
        processed data.

    """
    tt = pd.DataFrame()

    for seas in seasons:

        dfa = load_data(dbDir, dbName, runType, timescale, seas)

        ttb = process_season(dfa)
        tt = pd.concat((tt, ttb))

    tt['dbName'] = dbName

    outName = '{}/{}.hdf5'.format(outDir, dbName)
    tt.to_hdf(outName, key='zlim')
    del tt

    """
    tt = dfa.groupby(['field', 'healpixID', 'season']).apply(
        lambda x: get_zlim(x, sigmaC=sigmaC), include_groups=False).reset_index()

    tt['dbName'] = dbName
    """
    # return tt


def process_season(dfa):
    """
    Funtion to process a season

    Parameters
    ----------
    dfa : pandas df
        input data.

    Returns
    -------
    tt : pandas df
        processed data.

    """

    params = {}

    params['data'] = dfa
    hpixes = dfa['healpixID'].unique()

    tt = multiproc(hpixes, params, zlim_multiproc, nproc)

    return tt


def zlim_multiproc(toproc, params, j=0, output_q=None):
    """
    Estimate zlim using multiprocessing

    Parameters
    ----------
    toproc : list(str)
        List of healpixID to process.
    params : dict
        parameters for multiprocessing.
    j : int, optional
        internal int for multiproc. The default is 0.
    output_q : multiprocessing queue, optional
        where to put the results. The default is None.

    Returns
    -------
    pandas df
        Result.

    """

    data = params['data']

    idx = data['healpixID'].isin(toproc)

    dfa = data[idx]

    del data
    tt = dfa.groupby(['field', 'healpixID', 'season']).apply(
        lambda x: get_zlim(x, sigmaC=sigmaC), include_groups=False).reset_index()

    if output_q is not None:
        return output_q.put({j: tt})
    else:
        return tt


def get_zlim(grp, sigmaC=0.04):
    """
    Function to estimate the redshift limit corresponding to sigmaC >= 0.04

    Parameters
    ----------
    grp : pandas df
        Data to process.
    sigmaC : float, optional
        max sigmaC value for selection. The default is 0.04.

    Returns
    -------
    res : pandas df
        output resu.

    """

    dz = 0.05
    bins = np.arange(0.01, 1.1+dz, dz)
    df = bin_it_mean(grp, xvar='zmeas', yvar='sigmaC', bins=bins)

    df['sigmaC_plus'] = df['sigmaC']+df['sigmaC_std']
    df['sigmaC_minus'] = df['sigmaC']-df['sigmaC_std']

    zlim = interp1d(df['sigmaC'], df['zmeas'],
                    bounds_error=False, fill_value=0.)
    zlim_plus = interp1d(df['sigmaC_minus'], df['zmeas'],
                         bounds_error=False, fill_value=0.)
    zlim_minus = interp1d(df['sigmaC_plus'], df['zmeas'],
                          bounds_error=False, fill_value=0.)

    zlim = zlim(sigmaC)
    zlim_p = zlim_plus(sigmaC)
    zlim_m = zlim_minus(sigmaC)

    res = pd.DataFrame([zlim], columns=['zlim'])
    res['zlim_p'] = zlim_p
    res['zlim_m'] = zlim_m

    return res


parser = OptionParser(description='Script to analyze zlim for sigmaC<=0.04')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_zfaint_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='ddf_list.csv',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale [%default]')
parser.add_option('--sigmaC', type=float,
                  default=0.04,
                  help='sigma color max value for selection')
parser.add_option('--outDir', type=str,
                  default='../sn_zlim_sigmaC',
                  help='output directory')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='nproc for multiprocessing')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbList = opts.dbList
runType = opts.runType
timescale = opts.timescale
sigmaC = opts.sigmaC
outDir = opts.outDir
nproc = opts.nproc

# create outDir if necessary
checkDir(outDir)


# load dbList

df_list = pd.read_csv(dbList, comment='#')

# loop on data

for i, row in df_list.iterrows():
    process(dbDir, row['dbName'], runType, timescale, outDir, sigmaC, nproc)
