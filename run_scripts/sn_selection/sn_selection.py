#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:26:32 2023

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_analysis.sn_tools import load_complete_dbSimu, complete_df
from sn_analysis.sn_calc_plot import select
from sn_tools.sn_io import checkDir
from sn_analysis.sn_selection import selection_criteria
from optparse import OptionParser
import pandas as pd
import numpy as np
import glob


class Select_filt:
    def __init__(self, dataDir, dbName, sellist, seasons,
                 zType='spectroz', nsn_factor=1,
                 listFields='COSMOS', fieldType='DDF',
                 outDir='Test', nproc=8,
                 timescale='year',
                 dataType='pandasDataFrame',
                 ebvofMW=100):
        """
        class to load and select SN - output results: one file per season/year

        Parameters
        ----------
        dataDir : str
            Data directory.
        dbName : str
            Dbname to process.
        sellist : dict
            Selection criteria.
        seasons : list(int)
            seasons to process.
        zType : str, optional
            host z-type (spectroz/photz). The default is 'spectroz'.
        nsn_factor : int, optional
            nsn normalization factor. The default is 1.
        listFields : list(str), optional
            List of fields to process. The default is 'COSMOS'.
        fieldType : str, optional
            Type of field (DDF/WFD). The default is 'DDF'.
        outDir : str, optional
            Output dir. The default is 'Test'.
        nproc : int, optional
            Number of proc. The default is 8.
        timescale : str, optional
            Timescale (year/season). The default is 'year'.
        dataType : str, optional
            Data type. The default is 'pandasDataFrame'.
        ebvofMW : float, optional
            E(B-V) selection criteria. The default is 100.

        Returns
        -------
        None.

        """

        self.dataDir = dataDir
        self.dbName = dbName
        self.sellist = sellist
        self.seasons = seasons
        self.zType = zType
        self.nsn_factor = nsn_factor
        self.listFields = listFields
        self.fieldType = fieldType
        self.outDir = outDir
        self.nproc = nproc
        self.timescale = timescale
        self.dataType = dataType
        self.ebvofMW = ebvofMW

        self.outDir_full = self.init_dir()
        self.clean()

        self.process()

    def init_dir(self):
        """
        Method to create output dir

        Returns
        -------
        outDir_full : TYPE
            DESCRIPTION.

        """

        self.runType = '{}_{}'.format(self.fieldType, self.zType)
        outDir_full = '{}/{}/{}'.format(self.outDir, self.dbName, self.runType)
        checkDir(outDir_full)

        return outDir_full

    def clean(self):
        """
        Method to clean output dir

        Returns
        -------
        None.

        """

        import os

        for seas in self.seasons:
            outName = self.get_name(seas)

            if os.path.isfile(outName):
                os.remove(outName)

    def process(self):
        """
        Method to process data

        Returns
        -------
        None.

        """

        if self.fieldType == 'DDF':
            self.process_DDF()

        if self.fieldType == 'WFD':
            self.process_WFD()

    def process_DDF(self):
        """
        Method to process DDFs

        Returns
        -------
        None.

        """

        for seas in self.seasons:

            # load DDFs
            data = load_complete_dbSimu(
                self.dataDir, self.dbName, self.runType,
                listDDF=self.listFields, seasons=str(seas),
                nproc=self.nproc, dataType=self.dataType)
            print(seas, len(data))

            if data.empty:
                continue

            # E(B-V) cut
            idx = data['ebvofMW'] <= self.ebvofMW
            data = data[idx]

            # apply selection on Data
            sel_data = select(data, self.sellist)

            # get year
            sel_data = self.get_year(sel_data)

            # save the data
            self.save_data(sel_data, seas)

            # this is to get stat
            """
                stat, rname = get_stat(
                    sel_data, nsn_factor, timescale=timescale)
                stat[rname] = stat[rname].astype(int)
                stat_tot = pd.concat((stat_tot, stat))
                """

        # this is to get stat
        """
        if timescale == 'year':
            vv = ['nsn']+rname
            stat_tot = stat_tot.groupby(['field', timescale])[
                vv].sum().reset_index()

        stat_tot[timescale] = stat_tot[timescale].astype(int)
        stat_tot['nsn'] = stat_tot['nsn'].astype(int)
        outName_stat = '{}/nsn_{}_{}.hdf5'.format(
            outDir_full, dbName, timescale)
        store = pd.HDFStore(outName_stat, 'w')
        store.put('SN', stat_tot)

        # stat_tot.to_hdf(outName_stat, key='SN')
        """

    def process_WFD(self):
        """
        Method to process WFD

        Returns
        -------
        None.

        """

        deltaRA = 10.

        RAs = np.arange(0., 360.+deltaRA, deltaRA)

        RA_loop = []
        for RA in RAs[:-1]:
            RAmin = np.round(RA, 1)
            RAmax = RAmin+deltaRA
            RAmax = np.round(RAmax, 1)
            RA_loop.append((RAmin, RAmax))

        from sn_tools.sn_utils import multiproc
        params = {}
        multiproc(RA_loop, params, self.load_process, self.nproc)

    def load_process(self, toproc, params, j=0, output_q=None):
        """
        Method to load and process using multiprocessing

        Parameters
        ----------
        toproc : list((float, float))
            List of (RAmin, RAmax) to process.
        params : dict
            Parameters.
        j : int, optional
            internal int for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            Multiprocessing queue where to dump results. The default is None.

        Returns
        -------
        int
            Output data.

        """

        dfb = pd.DataFrame()
        for vv in toproc:
            self.load_process_RAs(vv[0], vv[1])
            # dfb = pd.concat((dfb, dd))
            """
            if len(dd) > 0:
                self.save_data_wfd(dd, vv[0], vv[1])
            """
        """
        if len(dfb) == 0:
            if output_q is not None:
                return output_q.put({j: 0})
            else:
                return 0

        for seas in self.seasons:
            idx = dfb['season'] == seas
            selb = dfb[idx]
            self.save_data(selb, seas)
        """
        if output_q is not None:
            return output_q.put({j: 0})
        else:
            return 0

    def save_data_wfd(self, sel_data, RAmin, RAmax):
        """
        Method to save WFD data

        Parameters
        ----------
        sel_data : pandas df
            Data to save.

        Returns
        -------
        None.

        """

        years = sel_data[self.timescale].unique()
        for vv in years:
            idx = sel_data[self.timescale] == vv
            selb = sel_data[idx]
            selb.to_hdf(self.get_name_wfd(vv, RAmin, RAmax),
                        key='SN', append=True)

    def load_process_RAs(self, RAmin, RAmax):
        """
        Method to load and process files


        Parameters
        ----------
        RAmin : float
            Min RA.
        RAmax : float
            Max RA.

        Returns
        -------
        None.

        """

        fullpath = '{}/{}/{}/*{}_{}*.hdf5'.format(self.dataDir, self.dbName,
                                                  self.runType, RAmin, RAmax)
        fis = glob.glob(fullpath)

        if len(fis) == 0:
            return pd.DataFrame()

        # load the data
        #data = pd.DataFrame()
        for fi in fis:
            print(fi)
            data = pd.read_hdf(fi)
            # estimate sigma_mu...
            if len(data) > 0:
                data = complete_df(data, alpha=0.4, beta=3)
            #data = pd.concat((data, dd))

            # E(B-V) cut
            idx = data['ebvofMW'] <= self.ebvofMW
            data = data[idx]
            # get year
            data = self.get_year(data)

            # apply selection on Data
            sel_data = select(data, self.sellist)

            if len(sel_data) > 0:
                self.save_data_wfd(sel_data, RAmin, RAmax)

        # return sel_data

        """
        for seas in self.seasons:
            idx = sel_data['season'] == seas
            selb = sel_data[idx]
            self.save_data(selb, seas)
        """

    def get_year(self, data):
        """
        Method to estimate the year
        and to add it as a df col

        Parameters
        ----------
        data : pandas df
            Data to process.

        Returns
        -------
        sel_data : pandas df
            original df plus year col.

        """

        sel_data = pd.DataFrame(data)
        sel_data = sel_data[sel_data.columns.drop(
            list(sel_data.filter(regex='mask')))]
        if 'selected' in sel_data.columns:
            sel_data = sel_data.drop(columns=['selected'])

        sel_data['year'] = 1
        if 'mjd_max' in sel_data.columns:
            tt = sel_data['mjd_max']-sel_data['lsst_start']
            # sel_data['year'] = sel_data['daymax']+60*(1.+sel_data['z'])
            # sel_data['year'] -= sel_data['lsst_start']
            tt /= 365.
            sel_data['year'] = np.ceil(tt)
            # print(sel_data['year'], sel_data['lsst_start'])
        sel_data['year'] = sel_data['year'].astype(int)
        # print(sel_data['year'])
        sel_data['chisq'] = sel_data['chisq'].astype(float)
        sel_data['sigmat0'] = np.sqrt(sel_data['Cov_t0t0'])
        sel_data['sigmax1'] = np.sqrt(sel_data['Cov_x1x1'])

        return sel_data

    def save_data(self, sel_data, seas):
        """
        Method to dump data on disk

        Parameters
        ----------
        sel_data : pandas df
            Data to dump.
        seas : int
            season.

        Returns
        -------
        None.

        """

        # save output data in pandas df
        if timescale == 'season':
            # store[seas].put('SN', sel_data)

            """
            outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
                self.outDir_full, self.fieldType, 
                self.dbName, self.timescale, seas)
            """
            sel_data.to_hdf(self.get_name(seas), key='SN')

        else:
            years = sel_data[self.timescale].unique()
            for vv in years:
                idx = sel_data[self.timescale] == vv
                selb = sel_data[idx]
                """
                outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
                    self.outDir_full, self.fieldType, self.dbName, self.timescale, vv)
                """
                selb.to_hdf(self.get_name(vv), key='SN', append=True)
                # store[vv].put('SN', selb)

    def get_name(self, seas):
        """
        Method to get the name of output files

        Parameters
        ----------
        seas : int
            season/year.

        Returns
        -------
        outName : str
            Output name.

        """

        outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
            self.outDir_full, self.fieldType,
            self.dbName, self.timescale, seas)

        return outName

    def get_name_wfd(self, seas, RAmin, RAmax):
        """
        Method to get the name of output files

        Parameters
        ----------
        seas : int
            season/year.

        Returns
        -------
        outName : str
            Output name.

        """

        outName = '{}/SN_{}_{}_{}_{}_{}_{}.hdf5'.format(
            self.outDir_full, self.fieldType,
            self.dbName, RAmin, RAmax, self.timescale, int(seas))

        return outName


def select_filt_deprecated(dataDir, dbName, sellist, seasons,
                           zType='spectroz', nsn_factor=1,
                           listFields='COSMOS', fieldType='DD',
                           outDir='Test', nproc=8,
                           timescale='year',
                           dataType='pandasDataFrame',
                           ebvofMW=100):
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
    ztype: str, opt
        redshift run type. The default is spectroz
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
    dataType: str, opt.
      data type to process. The default is 'pandasDataFrame'
     ebvofMW: float, opt.
        Max E(B-V). The default is 100.

    Returns
    -------
    None.

    """
    runType = '{}_{}'.format(fieldType, zType)
    outDir_full = '{}/{}/{}'.format(outDir, dbName, runType)
    checkDir(outDir_full)

    # remove files if necessary
    import os
    store = {}
    mydt = {}

    suffix = ''
    if fieldType == 'WFD':
        suffix = '_*'

    for seas in seasons:
        outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
            outDir_full, fieldType, dbName, timescale, seas)

        if os.path.isfile(outName):
            os.remove(outName)
        mydt[seas] = pd.DataFrame()

        # store[seas] = pd.HDFStore(outName, 'w')

    stat_tot = pd.DataFrame()
    # sel_tot = pd.DataFrame()

    process_DDF(dataDir, dbName, runType, listFields,
                seasons, nproc, dataType, suffix, outDir_full)


def process_DDF_deprecated(dataDir, dbName, runType, listFields,
                           seasons, nproc, dataType, suffix, outDir_full):

    for seas in seasons:

        # load DDFs
        data = load_complete_dbSimu(
            dataDir, dbName, runType, listDDF=listFields,
            seasons=str(seas), nproc=nproc, dataType=dataType, suffix=suffix)
        print(seas, len(data))

        if data.empty:
            continue

        # E(B-V) cut
        idx = data['ebvofMW'] <= ebvofMW
        data = data[idx]

        # apply selection on Data
        sel_data = select(data, sellist)

        # get year
        sel_data = get_year(sel_data)

        # save the data
        save_data(sel_data, outDir_full,
                  fieldType, dbName, timescale, seas)

        # this is to get stat
        """
            stat, rname = get_stat(sel_data, nsn_factor, timescale=timescale)
            stat[rname] = stat[rname].astype(int)
            stat_tot = pd.concat((stat_tot, stat))
            """

    # this is to get stat
    """
    if timescale == 'year':
        vv = ['nsn']+rname
        stat_tot = stat_tot.groupby(['field', timescale])[
            vv].sum().reset_index()

    stat_tot[timescale] = stat_tot[timescale].astype(int)
    stat_tot['nsn'] = stat_tot['nsn'].astype(int)
    outName_stat = '{}/nsn_{}_{}.hdf5'.format(outDir_full, dbName, timescale)
    store = pd.HDFStore(outName_stat, 'w')
    store.put('SN', stat_tot)

    # stat_tot.to_hdf(outName_stat, key='SN')
    """


def get_year_deprecated(data):
    """
    Function to estimate the year
    and to add it as a df col

    Parameters
    ----------
    data : pandas df
        Data to process.

    Returns
    -------
    sel_data : pandas df
        original df plus year col.

    """

    sel_data = pd.DataFrame(data)
    sel_data = sel_data[sel_data.columns.drop(
        list(sel_data.filter(regex='mask')))]
    if 'selected' in sel_data.columns:
        sel_data = sel_data.drop(columns=['selected'])

    sel_data['year'] = 1
    if 'mjd_max' in sel_data.columns:
        tt = sel_data['mjd_max']-sel_data['lsst_start']
        # sel_data['year'] = sel_data['daymax']+60*(1.+sel_data['z'])
        # sel_data['year'] -= sel_data['lsst_start']
        tt /= 365.
        sel_data['year'] = np.ceil(tt)
        # print(sel_data['year'], sel_data['lsst_start'])
    sel_data['year'] = sel_data['year'].astype(int)
    # print(sel_data['year'])
    sel_data['chisq'] = sel_data['chisq'].astype(float)

    return sel_data


def save_data_deprecated(sel_data, outDir_full,
                         fieldType, dbName, timescale, seas):
    """
    Function to dump data on disk

    Parameters
    ----------
    sel_data : pandas df
        Data to dump.
    outDir_full : str
        output dir.
    fieldType : str
        field type.
    dbName : str
        db name.
    timescale : str
        Timescale to use (year/season).
    seas : int
        season.

    Returns
    -------
    None.

    """

    # save output data in pandas df
    if timescale == 'season':
        # store[seas].put('SN', sel_data)

        outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
            outDir_full, fieldType, dbName, timescale, seas)

        sel_data.to_hdf(outName, key='SN')

    else:
        years = sel_data[timescale].unique()
        for vv in years:
            idx = sel_data[timescale] == vv
            selb = sel_data[idx]

            outName = '{}/SN_{}_{}_{}_{}.hdf5'.format(
                outDir_full, fieldType, dbName, timescale, vv)
            selb.to_hdf(outName, key='SN', append=True)
            # store[vv].put('SN', selb)


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


parser = OptionParser()

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell',
                  help="data dir[%default]")
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
parser.add_option("--nsn_factor", type=int,
                  default=30, help="MC normalisation factor [%default]")
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

opts, args = parser.parse_args()


dataDir = opts.dataDir
dbName = opts.dbName
selconfig = opts.selconfig
zType = opts.zType
fieldType = opts.fieldType
listFields = opts.listFields
nsn_factor = opts.nsn_factor
timescale = opts.timescale
dataType = opts.dataType
ebvofMW = opts.ebvofMW

outDir = '{}_{}'.format(dataDir, selconfig)
nproc = opts.nproc


seasons = range(1, 13)

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
            zType=zType, nsn_factor=nsn_factor,
            listFields=listFields, fieldType=fieldType,
            outDir=outDir, nproc=nproc,
            timescale=timescale,
            dataType=dataType,
            ebvofMW=ebvofMW)
