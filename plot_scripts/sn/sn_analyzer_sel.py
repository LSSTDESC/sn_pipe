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
from sn_analysis import plt
from sn_analysis.sn_calc_plot import bin_it, bin_it_mean
import h5py
from astropy.table import Table
from sn_tools.sn_utils import multiproc


def load_OS_table(dbDir, dbName, runType, season=1, fieldType='DDF',
                  years=[1], LSSTStart=60312):
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
        dfa = multiproc(keys, params, load_os_table_multi, nproc=16)

        if years:
            # add a year column
            dfa['year'] = (dfa['daymax']-LSSTStart)/365.+1.
            dfa = dfa.sort_values(by=['daymax'])

            dfa['year'] = dfa['year'].astype(int)

            # print(dfa[['year', 'daymax']], LSSTStart, years)

            idxb = dfa['year'].isin(years)
            dfa = dfa[idxb]

        # idx = dfa['ebvofMW'] < 0.25
        # dfa = dfa[idx]
        df = pd.concat((df, dfa))
        # break

    return df


def load_os_table_multi(keys, params, j=0, output_q=None):
    """
    Function to load a set of SN in astropy table format

    Parameters
    ----------
    keys : list(str)
        hdf5 keys.
    params : dict
        Parameters.
    j : int, optional
        tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        Where to put the result (if not None). The default is None.

    Returns
    -------
    dict or df
        Output.

    """

    df = pd.DataFrame()

    fFile = params['fFile']
    for key in keys:
        data = Table.read(fFile, path=key)
        df = pd.concat((df, data.to_pandas()))

    if output_q is not None:
        return output_q.put({j: df})
    else:
        return df


def load_OS_df(dbDir, dbName, runType, season=1, fieldType='DDF',
               years=[1], LSSTStart=60218):
    """


    Parameters
    ----------
    dbDir : TYPE
        DESCRIPTION.
    dbName : TYPE
        DESCRIPTION.
    runType : TYPE
        DESCRIPTION.
    season : TYPE, optional
        DESCRIPTION. The default is 1.
    fieldType : TYPE, optional
        DESCRIPTION. The default is 'DDF'.
    years : TYPE, optional
        DESCRIPTION. The default is [1].
    LSSTStart : TYPE, optional
        DESCRIPTION. The default is 60218.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
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
    years : list(int), optional
        Years to draw. The default is [1].
    LSSTStart : float, optional
        Start of LSST survey. The default is 60218.

    Returns
    -------
    df : pandas df
        OS data.

    """

    fullDir = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, runType)
    search_path = '{}/SN_{}_*_{}.hdf5'.format(fullDir, fieldType, season)

    print('search path', search_path, LSSTStart)

    fis = glob.glob(search_path)

    df = pd.DataFrame()

    for fi in fis:
        dfa = pd.read_hdf(fi)
        print('loading', fieldType, len(dfa))

        if years:
            # add a year column
            dfa['year'] = (dfa['daymax']-LSSTStart)/365.+1.
            dfa = dfa.sort_values(by=['daymax'])

            dfa['year'] = dfa['year'].astype(int)

            print(dfa[['year', 'daymax']], LSSTStart)

            # idxb = dfa['year'].isin(years)
            # dfa = dfa[idxb]

        # idx = dfa['ebvofMW'] < 0.25
        # dfa = dfa[idx]

        df = pd.concat((df, dfa))
        # break
    return df


def load_DataFrame(dbDir_WFD, OS_WFD, runType='spectroz',
                   seasons=[1],
                   years=[1], LSSTStart=60218., fieldType='WFD'):
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
    seasons: list(int), optional
        seasons to load. The default is [1].
    years : list(int), optional
        years to load. The default is [1].
    LSSTStart: float, optional.
      LSST start survey MJD. The default is 60218.

    Returns
    -------
    wfd : pandas df
        Loaded data.

    """

    if years:
        year_min = np.min(years)
        year_max = np.max(years)
        seas_min = np.max([1, year_min-2])
        seas_max = np.min([10, year_max+2])
        seasons = range(seas_min, seas_max+1)

    wfd = pd.DataFrame()
    for seas in seasons:
        print('loading season', seas, LSSTStart)
        wfd_seas = load_OS_df(dbDir_WFD, OS_WFD, runType=runType,
                              season=seas, fieldType=fieldType,
                              years=years, LSSTStart=LSSTStart)
        wfd_seas['dbName'] = OS_WFD
        wfd = pd.concat((wfd, wfd_seas))

    print(len(wfd))
    return wfd


def load_Table(dbDir_WFD, OS_WFD, runType='spectroz',
               seasons=[1],
               years=[1], LSSTStart=60218, fieldType='WFD'):
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
    seasons : list(int), optional
        seasons to load. The default is [1].
    years : list(int), optional
        years to load. The default is [1].
    LSSTStart : float, optional
        LSST MJD start. The default is 60218.

    Returns
    -------
    wfd : pandas df
        Loaded data.

    """

    if years:
        year_min = np.min(years)
        year_max = np.max(years)
        seas_min = np.max([1, year_min-2])
        seas_max = np.min([10, year_max+2])
        seasons = range(seas_min, seas_max+1)

    wfd = pd.DataFrame()
    for seas in seasons:
        print('loading season', seas)
        wfd_seas = load_OS_table(dbDir_WFD, OS_WFD, runType=runType,
                                 season=seas, fieldType=fieldType,
                                 years=years, LSSTStart=LSSTStart)
        wfd = pd.concat((wfd, wfd_seas))

    return wfd


def plot_DDF(data, norm_factor, nside=128):
    """
    Plot_nsn_vs(data, norm_factor, xvar='z', xleg='z',
                logy=True, cumul=True, xlim=[0.01, 1.1], nside=nside)

    Plot_nsn_vs(data, norm_factor, bins=np.arange(
        0.5, 11.5, 1), xvar='season', xleg='season',
        logy=False, xlim=[1, 10], nside=nside)
    """
    # plot_DDF_nsn(data, norm_factor, nside)

    # plot_DDF_dither(data, norm_factor, nside)

    plot_DDF_nsn_z(data, norm_factor, nside)

    """
    mypl.plot_nsn_versus_two(xvar='z', xleg='z', logy=True,
                             cumul=True, xlim=[0.01, 1.1])
    mypl.plot_nsn_mollview()
    """


def plot_DDF_nsn_z(data, norm_factor, nside):

    mypl = Plot_nsn_vs(data, norm_factor, nside)

    for field in data['field'].unique():
        idx = data['field'] == field
        sela = data[idx]
        fig, ax = plt.subplots(figsize=(14, 8))
        for dbName in sela['dbName'].unique():
            idxa = sela['dbName'] == dbName
            selb = sela[idxa]
            for season in selb['season'].unique():
                idxb = selb['season'] == season
                selc = selb[idxb]
                plot_nsn_versus_two(selc, xvar='z', xleg='z', logy=True,
                                    bins=np.arange(0.01, 1.15, 0.05),
                                    cumul=False, xlim=[0.01, 1.1],
                                    fig=fig, ax=ax, figtitle=field)


def plot_DDF_dither(data, norm_factor, nside):
    """
    Functio to plot and estimate dithering effects

    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : float
        Normalisation factor.
    nside : int
        Healpix nside parameter.

    Returns
    -------
    None.

    """

    mypl = Plot_nsn_vs(data, norm_factor, nside)

    nsn_pixels = data.groupby(['season', 'dbName', 'field',
                               'healpixID', 'pixRA', 'pixDec']
                              ).size().to_frame('nsn').reset_index()
    nsn_pixels['nsn'] /= norm_factor

    from sn_plotter_metrics.utils import get_dist

    nsn_pixels = nsn_pixels.groupby(['season', 'dbName', 'field']).apply(
        lambda x: get_dist(x)).reset_index()

    df_pixel = plot_field_season(nsn_pixels, mypl, xvar='dist', xleg='dist',
                                 yvar='nsn', yleg='$N_{SN}$', ls='None')

    npixels_FP = int(9.6 / pixelSize(nside))
    df_pixel.loc[:, 'nsn_no_dithering'] = npixels_FP*df_pixel['nsn_center']

    sums = get_sums_nsn(data, norm_factor, nside, cols=[
                        'season', 'dbName', 'field'])

    # merge wih sums to estimate the impact od the dithering
    sums = sums.merge(df_pixel, left_on=['season', 'dbName', 'field'],
                      right_on=['season', 'dbName', 'field'])

    print(sums)

    sums['nsn_loss_dither'] = 1. - (sums['nsn']/sums['nsn_no_dithering'])

    plot_field(sums, mypl, xvar='season', xleg='season',
               yvar='nsn_loss_dither', yleg='$N_{SN}$ loss [%]')

    plt.show()


def plot_DDF_nsn(data, norm_factor, nside):

    mypl = Plot_nsn_vs(data, norm_factor, nside)

    """
    mypl.plot_nsn_versus_two(xvar='z', xleg='z', logy=True,
                             cumul=True, xlim=[0.01, 1.1])
    mypl.plot_nsn_mollview()
    """

    # estimate the number of sn for all the fields/season

    sums = get_sums_nsn(data, norm_factor, nside, cols=[
                        'season', 'dbName', 'field'])
    sumt = get_sums_nsn(data, norm_factor, nside, cols=['season', 'dbName'])

    plot_field(sums, mypl)
    plot_field(sums, mypl, xvar='season', xleg='season',
               yvar='pixArea', yleg='Observed Area [deg$^{2}$]')

    # total number of SN per season/OS
    plot_field(sumt, mypl)
    plot_field(sumt, mypl, xvar='season', xleg='season',
               yvar='pixArea', yleg='Observed Area [deg$^{2}$]')
    plt.show()


def get_sums_nsn(data, norm_factor, nside, cols=['season', 'dbName', 'field']):
    """
    Function to estimate global parameters (nsn, pixArea, ...)

    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : float
        Normalization factor.
    nside : int
        healpix nside parameter.
    cols : list(str), optional
        Columns to make groups. The default is ['season', 'dbName', 'field'].

    Returns
    -------
    sums : pandas df
        Output data.

    """

    sums = data.groupby(cols).size().to_frame('nsn').reset_index()
    sums['nsn'] /= norm_factor

    if 'field' not in cols:
        sums['field'] = ','.join(data['field'].unique())
    pix = data.groupby(cols).apply(
        lambda x: pd.DataFrame({'npixels': [len(x['healpixID'].unique())]})).reset_index()

    pix['pixArea'] = pixelSize(nside)*pix['npixels']

    sums = sums.merge(pix, left_on=cols, right_on=cols)

    return sums


def pixelSize(nside):
    """
    Method to retuen the pixel size

    Parameters
    ----------
    nside : int
        nside healpix param.

    Returns
    -------
    pixSize : float
        pixel size.

    """

    import healpy as hp
    pixSize = hp.nside2pixarea(nside, degrees=True)

    return pixSize


def plot_field(data, mypl, xvar='season', xleg='season',
               yvar='nsn', yleg='$N_{SN}$'):
    """
    Function to plot a set of fields results

    Parameters
    ----------
    data : array
        Data to ptocess.
    mypl : class instance
        Plot_nsn_vs instance.
    xvar : str, optional
        x-axis variable. The default is 'season'.
    xleg : str, optional
        x-axis label. The default is 'season'.
    yvar : str, optional
        y-axis var. The default is 'nsn'.
    yleg : str, optional
        y-axis label. The default is '$N_{SN}$'.

    Returns
    -------
    None.

    """

    for field in data['field'].unique():
        idx = data['field'] == field
        sela = data[idx]
        fig, ax = plt.subplots(figsize=(14, 8))
        for dbName in sela['dbName'].unique():
            idxb = sela['dbName'] == dbName
            selb = sela[idxb]
            mypl.plot_versus(selb, xvar, xleg,
                             yvar, yleg,
                             figTitle=field, label=dbName, fig=fig, ax=ax, xlim=None)

        ax.legend()
        ax.grid()
        ax.set_xlabel(xleg)
        ax.set_ylabel(yleg)


def plot_field_season(data, mypl, xvar='dist', xleg='dist',
                      yvar='nsn', yleg='$N_{SN}$', ls='None'):
    """
    Function to plot a set of fields results

    Parameters
    ----------
    data : array
        Data to ptocess.
    mypl : class instance
        Plot_nsn_vs instance.
    xvar : str, optional
        x-axis variable. The default is 'season'.
    xleg : str, optional
        x-axis label. The default is 'season'.
    yvar : str, optional
        y-axis var. The default is 'nsn'.
    yleg : str, optional
        y-axis label. The default is '$N_{SN}$'.

    Returns
    -------
    None.

    """

    bins = np.arange(0.15, 2.15, 0.15)
    r = []

    for field in data['field'].unique():
        idx = data['field'] == field
        sela = data[idx]
        fig, ax = plt.subplots(figsize=(14, 8))
        for dbName in sela['dbName'].unique():
            idxb = sela['dbName'] == dbName
            selb = sela[idxb]
            for seas in selb['season'].unique():
                idxc = selb['season'] == seas
                selc = selb[idxc]
                seld = bin_it_mean(selc, xvar=xvar, yvar=yvar, bins=bins)
                seld = seld.fillna(-1.)
                idd = seld['nsn'] >= 0
                seld = seld[idd]
                mypl.plot_versus(seld, xvar, xleg,
                                 yvar, yleg,
                                 figTitle=field, label=None,
                                 fig=fig, ax=ax, xlim=None)

                idg = seld['dist'] <= 0.5
                nsn_mean = seld[idg]['nsn'].mean()
                r.append((field, dbName, seas, nsn_mean))

        ax.legend()
        ax.grid()
        ax.set_xlabel(xleg)
        ax.set_ylabel(yleg)

    res = pd.DataFrame(r, columns=['field', 'dbName', 'season', 'nsn_center'])

    return res


class Plot_nsn_vs:
    def __init__(self, data, norm_factor, nside=64):
        """
        class to plot ns vs z or season or ...

        Parameters
        ----------
        data : pandas df
            Data to plot.
        norm_factor : float
            Normalization factor.
        nside: int, optional
            nside healpix parameter. The default is 64.

        Returns
        -------
        None.

        """

        self.data = data
        self.norm_factor = norm_factor
        self.nside = nside

    def plot_versus(self, data, xvar='season', xleg='season',
                    yvar='nsn', yleg='$N_{SN}$', fig=None, ax=None,
                    figTitle='', label=None, xlim=[1, 10], ls='solid'):

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))

        fig.suptitle(figTitle)

        data = data.sort_values(by=[xvar])
        ax.plot(data[xvar], data[yvar], label=label, linestyle=ls, marker='.')
        ax.grid()
        if xlim is not None:
            ax.set_xlim(xlim)

    def plot_nsn_mollview(self, what='season'):
        """
        Method to plot the number of SN in Mollweid view

        Parameters
        ----------
        what : TYPE, optional
            DESCRIPTION. The default is 'season'.

        Returns
        -------
        None.

        """

        years = self.data[what].unique()

        self.Mollview_sum(self.data, addleg='all seasons')

        for year in years:
            idx = self.data[what] == year
            sel = self.data[idx]

            self.Mollview_sum(sel, addleg='{} {}'.format(what, int(year)))

        plt.show()

    def Mollview_sum(self, data, var='nsn', legvar='N$_{SN}$', addleg=''):
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
        xmin = 0.1
        xmax = xmax = np.max(sums[var])
        plotMollview(sums, var, legvar, addleg, np.sum,
                     xmin=xmin, xmax=xmax, nside=self.nside)


def plot_nsn_versus_two(data, norm_factor=30,
                        bins=np.arange(0.005, 0.8, 0.01),
                        xvar='z', xleg='z', logy=False,
                        cumul=False, xlim=[0.01, 0.8],
                        label='', fig=None, ax=None, figtitle=''):
    """
    Method to plot two curves sn vs ...

    Returns
    -------
    None.

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    fig.suptitle(figtitle)
    plot_nsn_binned(data, bins=bins, norm_factor=norm_factor,
                    xvar=xvar, xleg=xleg, logy=logy,
                    cumul=cumul, xlim=xlim,
                    label=label, fig=fig, ax=ax)
    idx = data['sigmaC'] <= 0.04
    label = '$\sigma_C \leq 0.04$'
    plot_nsn_binned(data[idx], norm_factor=norm_factor, bins=bins,
                    xvar=xvar, xleg=xleg, logy=logy,
                    cumul=cumul, xlim=xlim,
                    label=label, fig=fig, ax=ax)
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(xleg)
    ylabel = '$N_{SN}$'
    if cumul:
        ylabel = '$\sum N_{SN}$'
    ax.set_ylabel(r'{}'.format(ylabel))
    ax.legend()
    ax.grid()


def plot_nsn_binned(data, norm_factor=30,
                    bins=np.arange(0.005, 0.8, 0.01),
                    xvar='z', xleg='z', logy=False,
                    cumul=False, xlim=[0.01, 0.8],
                    label='', fig=None, ax=None):
    """
    Function to plot nsn vs...

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

    res = bin_it(data, xvar=xvar, bins=bins,
                 norm_factor=norm_factor)

    print(res)
    print('total number of SN', np.sum(res['NSN']))

    vv = res['NSN']
    if cumul:
        vv = np.cumsum(res['NSN'])
    ax.plot(res[xvar], vv, label=label)

    ax.set_xlabel(xleg)
    ax.set_ylabel(r'$N_{SN}$')
    ax.set_xlim(xlim)


def plotMollview(data, varName, leg, addleg, op, xmin, xmax, nside=128):
    """
    Function to display results as a Mollweid map

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
    nside: int, optional
        nside parameter for healpix. The default is 128

    """
    import healpy as hp
    npix = hp.nside2npix(nside)

    hpxmap = np.zeros(npix, dtype=np.float)
    hpxmap = np.full(hpxmap.shape, 0.)
    hpxmap[data['healpixID'].astype(
        int)] += data[varName]

    print(np.where(hpxmap < 0.01))

    norm = plt.cm.colors.Normalize(xmin, xmax)
    cmap = plt.cm.jet
    cmap.set_under('w')
    resleg = op(data[varName])
    if 'nsn' in varName:
        resleg = int(resleg)
    else:
        resleg = np.round(resleg, 2)
    title = '{}: {}'.format(leg, resleg)
    if addleg != '':
        title = '{} - {}'.format(addleg, title)

    hp.mollview(hpxmap, min=xmin, max=xmax, cmap=cmap,
                title=title, nest=True, norm=norm)
    hp.graticule()

    # save plot here
    name = leg.replace(' - ', '_')
    name = name.replace(' ', '_')

    # plt.savefig('Plots_pixels/Moll_{}.png'.format(name))


def get_val(var):
    """
    Function to grab values from parser

    Parameters
    ----------
    var : str
        var to process.

    Returns
    -------
    var : list(int)
        Result.

    """
    if '-' in var:
        seas_spl = var.split('-')
        seas_min = int(seas_spl[0])
        seas_max = int(seas_spl[1])
        var = range(seas_min, seas_max+1)
    else:
        var = var.split(',')
        var = list(map(int, var))

    return var


def process_WFD(conf_df, dataType, dbDir_WFD, runType, seasons,
                years, norm_factor):
    """
    Function to process WFD data

    Parameters
    ----------
    conf_df : pandas df
        Confituration file.
    dataType : str
        Data type.
    dbDir_WFD : str
        Location dir of WFD files.
    runType : str
        Type of run.
    seasons : list(int)
        Seasons to process.
    years : list(int)
        Years to process.
    norm_factor : float
        Normalization factor.

    Returns
    -------
    None.

    """

    OS_WFDs = conf_df['dbName_WFD'].unique()

    wfd = pd.DataFrame()
    for OS_WFD in OS_WFDs:
        idx = conf_df['dbName_WFD'] == OS_WFD
        LSSTStart = np.mean(conf_df[idx]['LSSTStart'])
        wfda = eval('load_{}(\'{}\',\'{}\',\'{}\',{},{},{})'.format(
            dataType, dbDir_WFD, OS_WFD, runType,
            seasons, years, LSSTStart))

        wfd = pd.concat((wfd, wfda))

    idx = wfd['ebvofMW'] < 0.25
    wfd = wfd[idx]

    plot_nsn_versus_two(wfd, xvar='season', xleg='season', logy=False,
                        bins=np.arange(0.5, 11.5, 1), norm_factor=norm_factor,
                        cumul=False, xlim=[1, 10])

    plot_nsn_versus_two(wfd, xvar='z', xleg='y', logy=True,
                        bins=np.arange(0.005, 0.805, 0.01), norm_factor=norm_factor,
                        cumul=True, xlim=[0.01, 0.8])

    """
    mypl = Plot_nsn_vs(wfd, norm_factor, nside=64)

    mypl.plot_nsn_mollview()
    """
    print(len(wfd))


def process_DDF(conf_df, dataType, dbDir_DD, runType, seasons, years, norm_factor):

    # load DDF
    OS_DDFs = conf_df['dbName_DD'].unique()

    ddf = pd.DataFrame()
    for OS_DDF in OS_DDFs:
        idx = conf_df['dbName_DD'] == OS_DDF
        LSSTStart = np.mean(conf_df[idx]['LSSTStart'])
        fieldType = 'DDF'
        ddfa = eval('load_{}(\'{}\',\'{}\',\'{}\',{},{},{},\'{}\')'.format(
            dataType, dbDir_DD, OS_DDF, runType,
            seasons, years, LSSTStart, fieldType))
        ddf = pd.concat((ddf, ddfa))

    plot_DDF(ddf, norm_factor, nside=128)


parser = OptionParser(description='Script to analyze SN prod after selection')

parser.add_option('--dbDir_DD', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--config', type=str,
                  default='config_ana.csv',
                  help='OS DD list[%default]')
parser.add_option('--norm_factor_DD', type=int,
                  default=30,
                  help='DD normalization factor [%default]')
parser.add_option('--dbDir_WFD', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
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
parser.add_option('--years', type=str,
                  default='1',
                  help='years to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')
# parser.add_option('--LSSTStart', type=float,
#                  default=60218.0018056,
#                  help='Survey starting date [%default]')


opts, args = parser.parse_args()

dbDir_DD = opts.dbDir_DD
norm_factor_DD = opts.norm_factor_DD
dbDir_WFD = opts.dbDir_WFD
norm_factor_WFD = opts.norm_factor_WFD
budget_DD = opts.budget_DD
runType = opts.runType
config = opts.config
seasons = opts.seasons

seasons = get_val(seasons)

years = opts.years
if years == 'None':
    years = []
else:
    years = get_val(years)

dataType = opts.dataType

# read config file
conf_df = pd.read_csv(config, comment='#')

# load wfds
"""
OS_WFDs = conf_df['dbName_WFD'].unique()

for OS_WFD in OS_WFDs:
    idx = conf_df['dbName_WFD'] == OS_WFD
    LSSTStart = np.mean(conf_df[idx]['LSSTStart'])
    wfd = eval('load_{}(\'{}\',\'{}\',\'{}\',{},{},{})'.format(
        dataType, dbDir_WFD, OS_WFD, runType,
        seasons, years, LSSTStart))

Plot_nsn_vs(wfd, norm_factor_DD, xvar='z', xleg='z',
            logy=True, cumul=True, xlim=[0.01, 0.7])

Plot_nsn_vs(wfd, norm_factor_DD, bins=np.arange(
    0.5, 11.5, 1), xvar='season', xleg='season', logy=False, xlim=[1, 10])

"""

process_WFD(conf_df, dataType, dbDir_WFD, runType,
            seasons, years, norm_factor_WFD)

# print(test)

# process_DDF(conf_df, dataType, dbDir_DD, runType, seasons, years,norm_factor_DD)

plt.show()
