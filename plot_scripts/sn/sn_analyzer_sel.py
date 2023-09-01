#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:50:54 2023

@author: philippe.gris@clermont.in2p3.fr
"""


import glob
from optparse import OptionParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sn_analysis.sn_calc_plot import bin_it
import h5py
from astropy.table import Table
from sn_tools.sn_utils import multiproc


def load_OS_table(dbDir, dbName, runType, season=1, fieldType='DDF'):
    """
    Function to load OS data

    Parameters
    ----------
    dbDir : str
        data location dir.
    dbName : str
        db name to process.
    runType : str
        run type (spectroz or photoz).
    season: int, optional
      season to process. The default is 1.
    fieldType : str, optional
        field type (DDF or WFD). The default is 'DDF'.

    Returns
    -------
    df : pandas df
        OS data.

    """

    fullDir = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, runType)
    search_path = '{}/SN_SN_{}_*_{}.hdf5'.format(fullDir, fieldType, season)

    fis = glob.glob(search_path)

    print('allo', fis, len(fis))

    df = pd.DataFrame()

    params = {}
    for fi in fis:
        fFile = h5py.File(fi, 'r')
        keys = list(fFile.keys())
        params['fFile'] = fFile
        dfa = multiproc(keys, params, load_os_table_multi, nproc=8)

        df = pd.concat((df, dfa))

    return df


def load_os_table_multi(keys, params, j=0, output_q=None):

    df = pd.DataFrame()

    fFile = params['fFile']
    for key in keys:
        data = Table.read(fFile, path=key)
        df = pd.concat((df, data.to_pandas()))

    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df


def load_OS_df(dbDir, dbName, runType, season=1, fieldType='DDF'):
    """
    Function to load OS data

    Parameters
    ----------
    dbDir : str
        data location dir.
    dbName : str
        db name to process.
    runType : str
        run type (spectroz or photoz).
    season: int, optional
      season to process. The default is 1.
    fieldType : str, optional
        field type (DDF or WFD). The default is 'DDF'.

    Returns
    -------
    df : pandas df
        OS data.

    """

    fullDir = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, runType)
    search_path = '{}/SN_{}_*_{}.hdf5'.format(fullDir, fieldType, season)

    fis = glob.glob(search_path)

    df = pd.DataFrame()

    for fi in fis:
        aa = pd.read_hdf(fi)
        print('loading', fieldType, len(aa))
        df = pd.concat((df, aa))
        # break
    return df


def load_DataFrame(dbDir_WFD, OS_WFD, runType='spectroz',
                   season=[1]):
    """
    Function to load data if pandas df

    Parameters
    ----------
    dbDir_WFD : str
        data location dir.
    OS_WFD : str
        WFD db name.
    runType : str, optional
        Run type. The default is spectroz.
    season : list(int), optional
        seasons to load. The default is [1].

    Returns
    -------
    wfd : pandas df
        Loaded data.

    """

    wfd = pd.DataFrame()
    for seas in seasons:
        wfd_seas = load_OS_df(dbDir_WFD, OS_WFD, runType=runType,
                              season=seas, fieldType='WFD')
        wfd = pd.concat((wfd, wfd_seas))

    return wfd


def load_Table(dbDir_WFD, OS_WFD, runType='spectroz',
               season=[1]):
    """
    Function to load data if pandas df

    Parameters
    ----------
    dbDir_WFD : str
        data location dir.
    OS_WFD : str
        WFD db name.
    runType : str, optional
        Run type. The default is spectroz.
    season : list(int), optional
        seasons to load. The default is [1].

    Returns
    -------
    wfd : pandas df
        Loaded data.

    """

    wfd = pd.DataFrame()
    for seas in seasons:
        wfd_seas = load_OS_table(dbDir_WFD, OS_WFD, runType=runType,
                                 season=seas, fieldType='WFD')
        wfd = pd.concat((wfd, wfd_seas))

    return wfd


class Plot_nsn_vs:
    def __init__(self, data, norm_factor, bins=np.arange(0.005, 0.8, 0.01),
                 xvar='z', xleg='z', logy=False, cumul=False, xlim=[0.01, 0.8],
                 nside=64):
        """
        class to plot ns vs z or season or ...

        Parameters
        ----------
        data : pandas df
            Data to plot.
        norm_factor : float
            Normalization factor.
        bins : numpy array, optional
            Bins for the plot. The default is np.arange(0.005, 0.8, 0.01).
        xvar : str, optional
            x-axis variable. The default is 'z'.
        xleg : str, optional
            x-axis label. The default is 'z'.
        logy : bool, optional
            To set the log scale for the y-axis. The default is False.
        cumul : bool, optional
            To plot cumulative plot. The default is False.
        xlim : list(float), optional
            x-axis limit. The default is [0.01, 0.8].
        nside: int, optional
            nside healpix parameter. The default is 64.

        Returns
        -------
        None.

        """

        self.data = data
        self.norm_factor = norm_factor
        self.bins = bins
        self.xvar = xvar
        self.xleg = xleg
        self.logy = logy
        self.cumul = cumul
        self.xlim = xlim
        self.nside = nside

        # self.plot_nsn_versus_two()
        self.plot_nsn_mollview()

    def plot_nsn_versus(self, data, label='', ax=None):
        """
        Method to plot nsn vs...

        Parameters
        ----------
        data : pandas df
            Data to plot.
        label : str, optional
            Curve label. The default is ''.
        ax : matplotlib axis, optional
            Axis for the plot. The default is None.

        Returns
        -------
        None.

        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))

        res = bin_it(data, xvar=self.xvar, bins=self.bins,
                     norm_factor=self.norm_factor)

        print(res)
        print('total number of SN', np.sum(res['NSN']))

        vv = res['NSN']
        if self.cumul:
            vv = np.cumsum(res['NSN'])
        ax.plot(res[self.xvar], vv, label=label)

        ax.set_xlabel(self.xleg)
        ax.set_ylabel(r'$N_{SN}$')
        ax.set_xlim(self.xlim)

    def plot_nsn_versus_two(self):
        """
        Method to plot two curves sn vs ...

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(figsize=(14, 8))
        self.plot_nsn_versus(self.data, ax=ax)
        idx = self.data['sigmaC'] <= 0.04
        label = '$\sigma_C \leq 0.04$'
        self.plot_nsn_versus(self.data[idx], label=label, ax=ax)
        if self.logy:
            ax.set_yscale("log")

        ax.set_xlabel(self.xleg)
        ylabel = '$N_{SN}$'
        if self.cumul:
            ylabel = '$\sum N_{SN}$'
        ax.set_ylabel(r'{}'.format(ylabel))
        ax.legend()
        ax.grid()

    def plot_nsn_mollview(self):

        seasons = self.data['season'].unique()

        for seas in seasons:
            idx = self.data['season'] == seas
            sel = self.data[idx]

            self.Mollview_sum(sel)

        plt.show()

    def Mollview_sum(self, data, var='nsn', legvar='NSN'):
        """
        Method to plot a Mollweid view for the sum of a variable

        Parameters
        --------------
        var: str,opt
          variable to show (default: nsn_zlim)
        legvar: str, opt
           name for title of the plot (default: NSN)

        """

        sums = data.groupby(['healpixID']).size().to_frame('nsn').reset_index()
        sums['nsn'] /= self.norm_factor
        print(sums)

        xmin = xmax = np.min(sums[var])
        xmin = 0.001
        xmax = xmax = np.max(sums[var])
        self.plotMollview(sums, var, legvar, np.sum,
                          xmin=xmin, xmax=xmax)

    def plotMollview(self, data, varName, leg, op, xmin, xmax):
        """
        Method to display results as a Mollweid map

        Parameters
        ---------------
        data: pandas df
          data to consider
        varName: str
          name of the variable to display
        leg: str
          legend of the plot
        op: operator
          operator to apply to the pixelize data(median, sum, ...)
        xmin: float
          min value for the display
        xmax: float
         max value for the display

        """
        import healpy as hp
        npix = hp.nside2npix(self.nside)

        hpxmap = np.zeros(npix, dtype=np.float)
        hpxmap = np.full(hpxmap.shape, 0.)
        hpxmap[data['healpixID'].astype(
            int)] += data[varName]

        norm = plt.cm.colors.Normalize(xmin, xmax)
        cmap = plt.cm.jet
        cmap.set_under('w')
        resleg = op(data[varName])
        if 'nsn' in varName:
            resleg = int(resleg)
        else:
            resleg = np.round(resleg, 2)
        title = '{}: {}'.format(leg, resleg)

        hp.mollview(hpxmap, min=xmin, max=xmax, cmap=cmap,
                    title=title, nest=True, norm=norm)
        hp.graticule()

        # save plot here
        name = leg.replace(' - ', '_')
        name = name.replace(' ', '_')

        # plt.savefig('Plots_pixels/Moll_{}.png'.format(name))


parser = OptionParser(description='Script to analyze SN prod after selection')

parser.add_option('--dbDir_DD', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList_DD', type=str,
                  default='input/DESC_cohesive_strategy/config_ana.csv',
                  help='OS DD list[%default]')
parser.add_option('--norm_factor_DD', type=int,
                  default=30,
                  help='DD normalization factor [%default]')
parser.add_option('--dbDir_WFD', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--OS_WFD', type=str,
                  default='draft_connected_v2.99_10yrs',
                  help='OS WFD [%default]')
parser.add_option('--norm_factor_WFD', type=int,
                  default=10,
                  help='WFD normalization factor [%default]')
parser.add_option('--budget_DD', type=float,
                  default=0.07,
                  help='DD budget [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='run type  [%default]')
parser.add_option('--seasons', type=str,
                  default='1',
                  help='seasons to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')

opts, args = parser.parse_args()

dbDir_DD = opts.dbDir_DD
dbList_DD = opts.dbList_DD
norm_factor_DD = opts.norm_factor_DD
dbDir_WFD = opts.dbDir_WFD
OS_WFD = opts.OS_WFD
norm_factor_WFD = opts.norm_factor_WFD
budget_DD = opts.budget_DD
runType = opts.runType
seasons = opts.seasons.split(',')
seasons = list(map(int, seasons))
dataType = opts.dataType


wfd = eval('load_{}(\'{}\',\'{}\',\'{}\',{})'.format(
    dataType, dbDir_WFD, OS_WFD, runType, seasons))
# Plot_nsn_vs(wfd, norm_factor_WFD, xvar='z', xleg='z',
#            logy=True, cumul=True, xlim=[0.01, 0.7])
Plot_nsn_vs(wfd, norm_factor_WFD, bins=np.arange(
    0.5, 11.5, 1), xvar='season', xleg='season', logy=False, xlim=[1, 10])

plt.show()
