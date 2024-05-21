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
from sn_analysis.sn_calc_plot import bin_it, bin_it_mean, bin_it_effi
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
        dfa = multiproc(keys, params, load_os_table_multi, nproc=16)

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


def load_OS_df(dbDir, dbName, runType, timescale_file='year',
               timeslot=1, fieldType='DDF'):
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
    timescale_file : str, optional
        Time scle of the files to load. The default is 'year'.
    timeslot : list(int), optional
        Time slots to process. The default is 1.
    fieldType : str, optional
        Field type to process. The default is 'DDF'.


    Returns
    -------
    df : pandas df
        OS data.

    """

    fullDir = '{}/{}/{}_{}'.format(dbDir, dbName, fieldType, runType)
    search_path = '{}/SN_{}_*_{}_{}.hdf5'.format(
        fullDir, fieldType, timescale_file, timeslot)

    print('search path', search_path)

    fis = glob.glob(search_path)

    df = pd.DataFrame()

    for fi in fis:
        dfa = pd.read_hdf(fi)
        print('loading', fieldType, len(dfa))

        # idx = dfa['ebvofMW'] < 0.25
        # dfa = dfa[idx]

        df = pd.concat((df, dfa))
        # break
    return df


def load_DataFrame(dbDir_WFD, OS_WFD, runType='spectroz',
                   timescale_file='year', timeslots=[1], fieldType='WFD'):
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
    timescale_file : str, optional
       Time scale of the files to process. The default is 'year'.
    timeslots : list(int), optional
       Time slots to process. The default is [1].
    fieldType : str, optional
       Field type to process. The default is 'WFD'.

    Returns
    -------
    wfd : pandas df
        Loaded data.

    """

    wfd = pd.DataFrame()
    for seas in timeslots:
        print('loading season', seas)
        wfd_seas = load_OS_df(dbDir_WFD, OS_WFD, runType=runType,
                              timescale_file=timescale_file,
                              timeslot=seas, fieldType=fieldType)
        wfd_seas['dbName'] = OS_WFD
        wfd = pd.concat((wfd, wfd_seas))

    print(len(wfd))

    # add a year column here
    # df_y = add_year(wfd, LSSTStart)

    return wfd


def add_year(wfd, LSSTStart):
    """
    Function to estimate the year SNe Ia have been observed

    Parameters
    ----------
    wfd : pandas df
        Data to process.
    LSSTStart : float
        LSST MJD start.

    Returns
    -------
    df_y : pandas df
        Original data + year col.

    """

    rf_phase = 35.
    df_y = pd.DataFrame()
    for y in range(1, 12):
        mjd_min = LSSTStart+(y-1)*365.
        mjd_max = LSSTStart+y*365.
        wfd['mjd_min'] = mjd_min-rf_phase*(1.+wfd['z'])
        wfd['mjd_max'] = mjd_max-rf_phase*(1.+wfd['z'])
        idx = wfd['daymax'] >= wfd['mjd_min']
        idx &= wfd['daymax'] < wfd['mjd_max']
        sel = pd.DataFrame(wfd[idx])
        sel['year'] = y
        df_y = pd.concat((df_y, sel))

    df_y = df_y.drop(columns=['mjd_min', 'mjd_max'])

    return df_y


def load_Table(dbDir_WFD, OS_WFD, runType='spectroz',
               seasons=[1], fieldType='WFD'):
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
    fieldType : str, optional
        Type of field to process. The default is 'WFD'.

    Returns
    -------
    wfd : pandas df
        Loaded data.

    """

    wfd = pd.DataFrame()
    for seas in seasons:
        print('loading season', seas)
        wfd_seas = load_OS_table(dbDir_WFD, OS_WFD, runType=runType,
                                 season=seas, fieldType=fieldType)

        wfd = pd.concat((wfd, wfd_seas))

    # add a year column here
    # df_y = add_year(wfd, LSSTStart)

    return wfd


def plot_DDF(data, norm_factor, config, nside=128, timescale='year'):
    """


    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : float
        Normalization factor.
    config: pandas df
      config for plot
    nside : int, optional
        Healpix nside parameter. The default is 128.
    timescale: str, opt
        Time scale for the plot. the default is 'year'

    Returns
    -------
    None.

    """
    """
    Plot_nsn_vs(data, norm_factor, xvar='z', xleg='z',
                logy=True, cumul=True, xlim=[0.01, 1.1], nside=nside)

    Plot_nsn_vs(data, norm_factor, bins=np.arange(
        0.5, 11.5, 1), xvar='season', xleg='season',
        logy=False, xlim=[1, 10], nside=nside)
    """
    # plot_DDF_nsn(data, norm_factor, config, nside, timescale=timescale)

    plot_survey_features(data, norm_factor, config, nside, timescale=timescale)

    # plot_DDF_dither(data, norm_factor, config, nside)

    # plot_DDF_nsn_z(data, norm_factor, nside)

    """
    mypl.plot_nsn_versus_two(xvar='z', xleg='z', logy=True,
                             cumul=True, xlim=[0.01, 1.1])
    mypl.plot_nsn_mollview()
    """


def plot_survey_features(data, norm_factor, config, nside, timescale):
    """
    Function to plot survey features related to sn

    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : int
        normalization factor.
    config : dict
        config params.
    nside : int
        nside param (healpix).
    timescale : str
        time scale to use (season/year).

    Returns
    -------
    None.

    """

    field = 'COSMOS'
    # dbName = 'baseline_v3.3_10yrs'
    dbName = 'DDF_DESC_0.80_WZ_0.07'

    plot_sn_features(data, field, dbName, timescale,
                     yvar='sigma_mu', ylabel='$\sigma_{\mu}}$ [mag]',
                     type_plot='sigma_mu', smoothIt=False)
    plot_sn_features(data, field, dbName, timescale,
                     yvar='NSN', ylabel='$N_{SN}$', type_plot='nsn',
                     smoothIt=True, norm_factor=norm_factor)

    plot_sn_features(data, field, dbName, timescale, smoothIt=True)

    plt.show()

    """
    df = get_zmax_field(data, field, dbName, timescale, zmin=0.7, sigmaC=1.e6)

    fig, ax = plt.subplots()
    ax.plot(df[timescale], df['nsn_zmin'], 'ko')

    plt.show()
    """


def plot_sn_features(data, field, dbName, timescale,
                     xvar='z', xlabel='$z$', yvar='sigma_mu',
                     ylabel='$frac^{N_{SN}}_{\sigma_{\mu} \leq \sigma_{int}}$',
                     yvar_cut=0.12, type_plot='effi', smoothIt=False,
                     norm_factor=1):
    """
    Function to plot sn features from survey

    Parameters
    ----------
    data : pandas df
        Data to process.
    field : str
        Field of interest.
    dbName : str
        OS name.
    timescale : str
        Time scale to use (season/year).
    xvar : str, optional
        x-axis var. The default is 'z'.
    xlabel : str, optional
        x-axis label. The default is '$z$'.
    yvar : str, optional
        y-axis var. The default is 'sigma_mu'.
    ylabel : str, optional
        y-axis label.
        The default is '$frac^{N_{SN}}_{\sigma_{\mu} \leq \sigma_{int}}$'.
    yvar_cut : float, optional
        y-axis selection cut. The default is 0.12.
    type_plot : str, optional
        type of plot (sigma_mu, nsn, effi). The default is 'effi'.
    smoothIt : bool, optional
        To smooth (spline) displayed curves. The default is False.
    norm_factor : int, optional
        normalization factor. The default is 1.

    Returns
    -------
    None.

    """

    idx = data['field'] == field
    idx &= data['dbName'] == dbName

    dbNameb = '_'.join(dbName.split('_')[:-1])
    sel = data[idx]

    # for each year: sigmamu vs z
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(right=0.82)
    fig.suptitle('{} - {}'.format(dbNameb, field))
    ttimes = range(1, 12)
    lls = ['solid']*4+['dashed']*4+['dotted']*4
    mmarkers = ['o', '*', '^', 'h']*3
    listy = dict(zip(ttimes, lls))
    marks = dict(zip(ttimes, mmarkers))

    for timeslot in timeslots:

        idxb = sel[timescale] == timeslot
        selb = sel[idxb]

        eval('plot_{}(ax, selb, xvar, yvar, yvar_cut, smoothIt,marks, listy, timescale, timeslot, norm_factor)'.format(type_plot))

        """
        if type_plot == 'sigma_mu':
            plot_sigma_mu(ax, selb, xvar, yvar, yvar_cut, smoothIt,
                          marks, listy, timescale, timeslot)
        if type_plot == 'nsn':
            plot_nsn(ax, selb, xvar, yvar, yvar_cut, smoothIt,
                     marks, listy, timescale, timeslot, norm_factor)

        if type_plot == 'effi':
            plot_effi(ax, selb, xvar, yvar, yvar_cut, smoothIt,
                      marks, listy, timescale, timeslot)
        """
    ax.grid(visible=True)
    ax.set_xlabel(r'{}'.format(xlabel))
    ax.set_ylabel(r'{}'.format(ylabel))

    ax.legend(loc='upper center',
              bbox_to_anchor=(1.12, 0.7),
              ncol=1, fontsize=15, frameon=False)


def plot_sigma_mu(ax, selb, xvar, yvar, yvar_cut,
                  smoothIt, marks, listy, timescale, timeslot, norm_factor):
    """
    Function to plot sigma_mu vs z

    Parameters
    ----------
    ax : matplotlib axis
        plot axis.
    selb : pandas df
        Data to plot.
    xvar : str
        x-axis var.
    yvar : str
        y-axis var.
    yvar_cut : float
        y var selection cut.
    smoothIt : bool
        To smooth (spline) displayed curves.
    marks : dict
        Markers used for the plot.
    listy : dict
        linestyle used for the plot.
    timescale : str
        time scale to use (season/year).
    timeslot : int
        time slot for display.
    norm_factor : int
        Normalization factor.
    Returns
    -------
    None.

    """

    df = bin_it_mean(selb, xvar=xvar, yvar=yvar,
                     bins=np.arange(0.01, 1.12, 0.02))
    # ax.errorbar(df['z'], df['sigma_mu'], yerr=df['sigma_mu_std'])
    ax.plot(df[xvar], df[yvar], color='k', marker=marks[timeslot],
            ls=listy[timeslot], mfc='None', ms=10, markevery=5,
            label='{} {}'.format(timescale, timeslot))
    xmin = 0.2
    xmax = 1.08
    ymin = 0.0
    ymax = 0.6
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.plot([xmin, xmax], [0.12]*2, ls='dashed', color='r')
    ttext = '$\sigma_{int}\sim$0.12'
    ax.text(0.3, 0.13, ttext, color='r')


def plot_nsn(ax, selb, xvar, yvar, yvar_cut,
             smoothIt, marks, listy, timescale, timeslot, norm_factor):
    """
    Function to plot nsn vs z

    Parameters
    ----------
    ax : matplotlib axis
        plot axis.
    selb : pandas df
        Data to plot.
    xvar : str
        x-axis var.
    yvar : str
        y-axis var.
    yvar_cut : float
        y var selection cut.
    smoothIt : bool
        To smooth (spline) displayed curves.
    marks : dict
        Markers used for the plot.
    listy : dict
        linestyle used for the plot.
    timescale : str
        time scale to use (season/year).
    timeslot : int
        time slot for display.
    norm_factor : int
        Normalization factor.

    Returns
    -------
    None.

    """

    df = bin_it(selb, xvar=xvar, norm_factor=norm_factor,
                bins=np.arange(0.01, 1.12, 0.05))
    # ax.errorbar(df['z'], df['sigma_mu'], yerr=df['sigma_mu_std'])
    if smoothIt:
        from scipy.interpolate import make_interp_spline
        xnew = np.linspace(
            np.min(df[xvar]), np.max(df[xvar]), 100)
        spl = make_interp_spline(
            df[xvar], df[yvar], k=3)  # type: BSpline
        spl_smooth = spl(xnew)

        ax.plot(xnew, spl_smooth, color='k',
                marker=marks[timeslot], ls=listy[timeslot],
                mfc='None', ms=10, markevery=5,
                label='{} {}'.format(timescale, timeslot))

    else:
        ax.plot(df[xvar], df[yvar], color='k',
                marker=marks[timeslot], ls=listy[timeslot],
                mfc='None', ms=10, markevery=5,
                label='{} {}'.format(timescale, timeslot))

    xmin = 0.2
    xmax = 1.08
    ymin = 0.0
    ymax = None
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])


def plot_effi(ax, selb, xvar, yvar, yvar_cut,
              smoothIt, marks, listy, timescale, timeslot, norm_factor):
    """
    Function to plot effi vs z

    Parameters
    ----------
    ax : matplotlib axis
        plot axis.
    selb : pandas df
        Data to plot.
    xvar : str
        x-axis var.
    yvar : str
        y-axis var.
    yvar_cut : float
        y var selection cut.
    smoothIt : bool
        To smooth (spline) displayed curves.
    marks : dict
        Markers used for the plot.
    listy : dict
        linestyle used for the plot.
    timescale : str
        time scale to use (season/year).
    timeslot : int
        time slot for display.
    norm_factor : int
        Normalization factor.

    Returns
    -------
    None.

    """

    df = bin_it_effi(selb, xvar=xvar, yvar=yvar, yvar_cut=yvar_cut,
                     bins=np.arange(0.01, 1.12, 0.05))

    print(df)
    # ax.errorbar(df['z'], df['sigma_mu'], yerr=df['sigma_mu_std'])

    if smoothIt:
        from scipy.interpolate import make_interp_spline
        xnew = np.linspace(
            np.min(df[xvar]), np.max(df[xvar]), 100)
        spl = make_interp_spline(
            df[xvar], df['effi'], k=3)  # type: BSpline
        spl_smooth = spl(xnew)

        ax.plot(xnew, spl_smooth, color='k',
                marker=marks[timeslot], ls=listy[timeslot],
                mfc='None', ms=10, markevery=5,
                label='{} {}'.format(timescale, timeslot))

    else:
        ax.plot(df[xvar], df['effi'], color='k',
                marker=marks[timeslot], ls=listy[timeslot],
                mfc='None', ms=10, markevery=5,
                label='{} {}'.format(timescale, timeslot))
    xmin = 0.2
    xmax = 1.08
    ymin = 0.0
    ymax = None
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    print(df)
    ax.plot([xmin, xmax], [0.95]*2, ls='dashed', color='r')
    ttext = '0.95'
    ax.text(0.3, 0.92, ttext, color='r', fontsize=10)


def get_zmax_field(data, field, dbName, timescale, zmin, sigmaC):

    idx = data['field'] == field
    idx &= data['dbName'] == dbName

    sel = data[idx]

    # for each year: cumulative vs z

    timeslots = sel[timescale].unique()

    r = []
    for timeslot in timeslots:

        idxb = sel[timescale] == timeslot
        idxb &= sel['sigmaC'] <= sigmaC
        selb = sel[idxb]

        nsntot = len(selb)
        idxc = selb['z'] >= zmin
        selc = selb[idxc]
        nsn_zmin = len(selc)
        frac = nsn_zmin/nsntot
        print(timeslot, zmin, frac)
        r.append((dbName, field, timeslot, zmin, frac, nsn_zmin))

    cols = ['dbName', 'field', timescale, 'zmin', 'frac', 'nsn_zmin']

    df = pd.DataFrame(r, columns=cols)

    return df


def plot_DDF_nsn_z(data, norm_factor, nside, timescale='year'):
    """
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

    # mypl = Plot_nsn_vs(data, norm_factor, nside)

    for field in data['field'].unique():
        idx = data['field'] == field
        sela = data[idx]

        for dbName in sela['dbName'].unique():
            idxa = sela['dbName'] == dbName
            selb = sela[idxa]
            fig, ax = plt.subplots(figsize=(14, 8))
            for season in selb[timescale].unique():
                idxb = selb[timescale] == season
                idxb &= selb['sigma_mu'] <= 0.12
                selc = selb[idxb]

                plot_nsn_binned(selc, xvar='z', xleg='z', logy=True,
                                bins=np.arange(0.01, 1.15, 0.1),
                                cumul=False, xlim=[0.01, 1.1],
                                fig=fig, ax=ax, figtitle='{} - {}'.format(
                                    dbName, field))


def plot_DDF_dither(data, norm_factor, config, nside, timescale='year'):
    """
    Functio to plot and estimate dithering effects

    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : float
        Normalisation factor.
    config: pandas df
       config for plot.
    nside : int
        Healpix nside parameter.
    timescale: str, optional.
    Time scale for estimation. The default is 'year'

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

    plot_field(sums, mypl, xvar=timescale, xleg=timescale,
               yvar='nsn_loss_dither', yleg='$N_{SN}$ loss [%]')

    plt.show()


def plot_DDF_nsn(data, norm_factor, config, nside, sigmaC=1.e6, timescale='year'):
    """


    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : float
        Normalization factor
    config: pandas df
      config for plots.
    nside : int
        Healpix nside parameter.
    sigmaC: float, optional.
     sigmaC selection cut. The default is 1.e6
    timescale: str, opt
    Time scale for the plot. The default is 'year'

    Returns
    -------
    None.

    """

    idx = data['sigmaC'] <= sigmaC

    data = data[idx]

    mypl = Plot_nsn_vs(data, norm_factor, nside)
    # mypl.plot_nsn_mollview()
    """
    # mypl.plot_nsn_versus_two(xvar='z', xleg='z', logy=True,
    #                         cumul=True, xlim=[0.01, 1.1])
    mypl.plot_nsn_mollview()
    """

    # estimate the number of sn for all the fields/season

    sums = get_sums_nsn(data, norm_factor, nside, cols=[
        timescale, 'dbName', 'field'])
    sumt = get_sums_nsn(data, norm_factor, nside, cols=[timescale, 'dbName'])

    plot_field(sums, mypl, config, xvar=timescale,
               xleg=timescale, cumul=True)
    # plot_field(sums, mypl, xvar=timescale, xleg=timescale,
    #           yvar='pixArea', yleg='Observed Area [deg$^{2}$]')

    # total number of SN per season/OS
    plot_field(sumt, mypl, config, xvar=timescale, xleg=timescale,
               cumul=True)
    # plot_field(sumt, mypl, xvar=timescale, xleg=timescale,
    #           yvar='pixArea', yleg='Observed Area [deg$^{2}$]')
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


def plot_field(data, mypl, config, xvar='season', xleg='season',
               yvar='nsn', yleg='$N_{SN}$', cumul=False, norm='', logy=False):
    """
    Function to plot a set of fields results

    Parameters
    ----------
    data : array
        Data to ptocess.
    mypl : class instance
        Plot_nsn_vs instance.
    config: pandas df
      config for plots
    xvar : str, optional
        x-axis variable. The default is 'season'.
    xleg : str, optional
        x-axis label. The default is 'season'.
    yvar : str, optional
        y-axis var. The default is 'nsn'.
    yleg : str, optional
        y-axis label. The default is '$N_{SN}$'.
    cumul : bool, optional
        for cumulative plot. The default is False.
    Returns
    -------
    None.

    """

    if norm != '':
        # normalize the results here
        idx = data['dbName'] == norm
        selnorm = data[idx]
        vmerge = ['field', xvar]
        df = data.merge(selnorm, left_on=vmerge, right_on=vmerge)
        df['{}'.format(yvar)] = df['{}_x'.format(yvar)]/df['{}_y'.format(yvar)]
        df['dbName'] = df['dbName_x']
        data = pd.DataFrame(df)

    for field in data['field'].unique():
        idx = data['field'] == field
        sela = data[idx]
        fig, ax = plt.subplots(figsize=(14, 8))
        for dbName in sela['dbName'].unique():
            idxb = sela['dbName'] == dbName
            selb = sela[idxb]
            idxc = config['dbName_DD'] == dbName
            conf = config[idxc]
            ls = conf['ls'].to_list()[0]
            color = conf['color'].to_list()[0]
            marker = conf['marker'].to_list()[0]
            mypl.plot_versus(selb, xvar, xleg,
                             yvar, yleg,
                             figTitle=field, label=dbName,
                             fig=fig, ax=ax, xlim=None, cumul=cumul,
                             ls=ls, color=color,
                             marker=marker)

        ax.legend()
        # ax.grid()
        ax.set_xlabel(xleg, fontweight='bold')
        ax.set_ylabel(yleg, fontweight='bold')
        ax.grid(visible=True)
        if logy:
            ax.set_yscale("log")


def plot_field_time(data, mypl, config, xvar='dist', xleg='dist',
                    yvar='nsn', yleg='$N_{SN}$', ls='None', timescale='year'):
    """
    Function to plot a set of fields results

    Parameters
    ----------
    data : array
        Data to ptocess.
    mypl : class instance
        Plot_nsn_vs instance.
    config: pandas df
      config for plots
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
        fig, ax = plt.subplots(figsize=(14, 9))
        for dbName in sela['dbName'].unique():
            idxb = sela['dbName'] == dbName
            selb = sela[idxb]
            idxcc = config['dbName_DD'] == dbName
            conf = config[idxcc]
            ls = conf['ls'].to_list()[0]
            color = conf['color'].to_list()[0]
            marker = conf['marker'].to_list()[0]
            for seas in selb[timescale].unique():
                idxc = selb[timescale] == seas
                selc = selb[idxc]
                seld = bin_it_mean(selc, xvar=xvar, yvar=yvar, bins=bins)
                seld = seld.fillna(-1.)
                idd = seld['nsn'] >= 0
                seld = seld[idd]
                mypl.plot_versus(seld, xvar, xleg,
                                 yvar, yleg,
                                 figTitle=field, label=None,
                                 fig=fig, ax=ax, xlim=None,
                                 ls=ls, color=color,
                                 marker=marker)

                idg = seld['dist'] <= 0.5
                nsn_mean = seld[idg]['nsn'].mean()
                r.append((field, dbName, seas, nsn_mean))

        ax.legend()
        ax.grid()
        ax.set_xlabel(xleg)
        ax.set_ylabel(yleg)

    res = pd.DataFrame(r, columns=['field', 'dbName', timescale, 'nsn_center'])

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
                    figTitle='', label=None, xlim=[1, 10],
                    ls='solid', cumul=False, color='k', marker='o'):

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 9))

        fig.suptitle(figTitle)

        data = data.sort_values(by=[xvar])
        datab = data[yvar]
        if cumul:
            datab = np.cumsum(datab)
        ax.plot(data[xvar], datab, label=label,
                linestyle=ls, marker=marker, color=color, mfc='None', lw=3)
        ax.grid()
        if xlim is not None:
            ax.set_xlim(xlim)

    def plot_nsn_mollview(self, what='season', dbName=''):
        """
        Method to plot the number of SN in Mollweid view

        Parameters
        ----------
        what : TYPE, optional
            DESCRIPTION. The default is 'season'.
        dbName: str, optional
          dbName to display. The default is ''

        Returns
        -------
        None.

        """

        years = self.data[what].unique()

        saveName = '{}_moll'.format(dbName)
        self.Mollview_sum(self.data, addleg='{}'.format(
            dbName), saveName=saveName)

        for year in years:
            idx = self.data[what] == year
            sel = self.data[idx]

            saveName = '{}_moll_{}'.format(dbName, year)
            self.Mollview_sum(sel, addleg='{} \n {} {}'.format(dbName, what, int(year)),
                              saveName=saveName)

        # plt.show()

    def Mollview_sum(self, data, var='nsn',
                     legvar='N$_{SN}$', addleg='', saveName=''):
        """
        Method to plot a Mollweid view for the sum of a variable
        Parameters
        ----------
        data : pandas df
            Data to plot.
        var : str, optional
            Variable to display. The default is 'nsn'.
        legvar : str, optional
            plot legend. The default is 'N$_{SN}$'.
        addleg : str, optional
            Additionnal info for legend. The default is ''.
        saveName : str, optional
            name for the jpeg file. The default is ''.

        Returns
        -------
        None.

        """

        sums = data.groupby(['healpixID']).size().to_frame('nsn').reset_index()
        sums['nsn'] /= self.norm_factor
        print(sums)

        xmin = xmax = np.min(sums[var])
        xmin = 0.1
        xmax = xmax = np.max(sums[var])
        plotMollview(sums, var, legvar, addleg, np.sum,
                     xmin=xmin, xmax=xmax,
                     nside=self.nside, saveName=saveName)


def plot_nsn_versus_two(data, norm_factor=30, nside=128,
                        bins=np.arange(0.005, 0.8, 0.01),
                        xvar='z', xleg='z', logy=False,
                        cumul=False, xlim=[0.01, 0.8],
                        label='', fig=None, ax=None, figtitle='',
                        color='k', marker='o', cumnorm=False):
    """
    Method to plot two curves sn vs ...

    Returns
    -------
    None.

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    fig.suptitle(figtitle)

    # idxa = data['z'] <= 0.1
    # data = data[idxa]

    label = data['dbName'].unique()[0]
    print('rrr', xvar)
    plot_nsn_binned(data, bins=bins, norm_factor=norm_factor,
                    nside=nside,
                    xvar=xvar, xleg=xleg, logy=logy,
                    cumul=cumul, xlim=xlim,
                    label=label, fig=fig, ax=ax, color=color, marker=marker,
                    cumnorm=cumnorm)
    idx = data['sigma_mu'] <= 0.12
    labelb = label+' - $\sigma_{\mu} \leq 0.12$'
    plot_nsn_binned(data[idx], norm_factor=norm_factor, bins=bins,
                    xvar=xvar, xleg=xleg, logy=logy,
                    cumul=cumul, xlim=xlim,
                    label=labelb, fig=fig, ax=ax, ls='dotted',
                    color=color, marker=marker, cumnorm=cumnorm)
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(xleg, fontweight='bold')
    ylabel = '$N_{SN}$'
    if cumul:
        ylabel = '$\sum N_{SN}$'
    ax.set_ylabel(r'{}'.format(ylabel), fontweight='bold')
    ax.legend()
    ax.grid()


def plot_versus(df, xvar='year', xlabel='year',
                yvar='nsn', ylabel='$N_{SN}$', fig=None, ax=None,
                label='', ls='solid', marker='o', color='k', cumul=False, mfc='k'):

    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    ypl = df[yvar]
    if cumul:
        ypl = np.cumsum(ypl)

    ax.plot(df[xvar], ypl, ls=ls, marker=marker,
            color=color, label=label, mfc=mfc, markersize=9, lw=2)


def plot_nsn_binned(data, norm_factor=30, nside=128,
                    bins=np.arange(0.005, 0.8, 0.01),
                    xvar='z', xleg='z', logy=False,
                    cumul=False, xlim=[0.01, 0.8],
                    label='', fig=None, ax=None, color='k',
                    ls='solid', figtitle='', marker='o', frac=0.95,
                    cumnorm=False):
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

    fig.suptitle(figtitle)

    res = bin_it(data, xvar=xvar, bins=bins,
                 norm_factor=norm_factor)

    nsn_tot = np.sum(res['NSN'])
    print('total number of SN', nsn_tot)

    npixels = len(data['healpixID'].unique())
    import healpy as hp
    pixArea = hp.nside2pixarea(nside, degrees=True)

    print('density', nsn_tot/(npixels*pixArea))

    vv = res['NSN']
    if cumul:
        vv = np.cumsum(res['NSN'])
        print(vv)
        if cumnorm:
            vv /= vv.max()
    if label != '':
        ax.plot(res[xvar], vv, label=label, color=color,
                linestyle=ls, marker=marker, markersize=9, lw=2)
    else:
        ax.plot(res[xvar], vv, color=color, linestyle=ls,
                marker=marker, markersize=9, lw=2)

    ax.set_xlabel(xleg)
    ax.set_ylabel(r'$N_{SN}$', fontweight='bold')
    ax.set_xlim(xlim)

    if cumul:
        from scipy.interpolate import interp1d
        vv = interp1d(vv, res[xvar], bounds_error=False, fill_value=0.)
        print(vv(frac))


def plotMollview(data, varName, leg, addleg, op, xmin, xmax,
                 nside=128, saveName=''):
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
    saveName:str, optional.
       output name for the jpeg. The default is ''

    """
    import healpy as hp
    npix = hp.nside2npix(nside)

    fig = plt.figure(figsize=(8, 6))

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

    hp.mollview(hpxmap, fig=fig, min=xmin, max=xmax, cmap=cmap,
                title=title, nest=True, norm=norm)
    hp.graticule()

    # save plot here
    name = leg.replace(' - ', '_')
    name = name.replace(' ', '_')

    if saveName != '':
        plt.savefig('Plots_pixels/{}.png'.format(saveName))


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


def process_WFD_OS(conf_df, dataType, dbDir_WFD, runType,
                   timescale_file, timeslots, norm_factor, nside, outName,
                   plot_moll=False):
    """
    Function to process WFD data

    Parameters
    ----------
    conf_df : pandas df
        config file.
    dataType : str
        Data type.
    dbDir_WFD : str
        Data dir.
    runType : str
        Run type.
    timescale_file : str
        Time scale (year/season)
    timeslots : list(int)
        Time slots
    norm_factor : float
        Normalization factor
    nside : int
        Healpix nside parameter.
    outName : str
        Output name.
    plot_moll : bool, optional
        To plot Mollview. The default is False.

    Returns
    -------
    wfd : pandas df
        Output data.

    """

    OS_WFDs = conf_df['dbName_WFD'].unique()
    wfd = pd.DataFrame()
    fig, ax = plt.subplots(figsize=(14, 8))
    for OS_WFD in OS_WFDs:
        idx = conf_df['dbName_WFD'] == OS_WFD
        tt = 'load_{}(\'{}\',\'{}\',\'{}\',\'{}\',{})'.format(
            dataType, dbDir_WFD, OS_WFD, runType,
            timescale_file, timeslots)
        wfda = eval(tt)
        idx = wfda['ebvofMW'] < 0.25
        wfda = wfda[idx]
        # plot mollview here
        if plot_moll:
            mypl = Plot_nsn_vs(wfda, norm_factor, nside=64)
            mypl.plot_nsn_mollview(what=timescale_file, dbName=OS_WFD)
        idc = conf_df['dbName_WFD'] == OS_WFD
        selp = conf_df[idc]
        ls = selp['ls'].values[0]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        """
        plot_nsn_versus_two(wfda, xvar='z', xleg='z', logy=False,
                            bins=np.arange(0.005, 0.805, 0.05),
                            norm_factor=norm_factor,
                            cumul=False, xlim=[0.01, 0.8], color=color,
                            marker=marker, fig=fig, ax=ax, cumnorm=True)
        """
        # density plot here
        # Plot_density(wfda, timescale_file, OS_WFD)

        # wfd = pd.concat((wfd, wfda))
        # get some stat

        wfda = wfda.groupby(['dbName', timescale_file]).apply(
            lambda x: get_stat(x, norm_factor)).reset_index()
        wfd = pd.concat((wfd, wfda))

    ax.legend(loc='upper left', bbox_to_anchor=(
        0.0, 1.15), ncol=3, fontsize=11, frameon=False)
    plt.show()

    # print(wfd)

    # wfd.to_csv(outName, index=False)

    print('out')
    return wfd


def process_WFD(conf_df, dataType, dbDir_WFD, runType,
                timescale_file, timeslots, norm_factor,
                nside=64, timescale='season', plot_moll=False):
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
    norm_factor : float
        Normalization factor.

    Returns
    -------
    None.

    """

    outName = 'nsn_WFD_v3_test.csv'

    wfd = process_WFD_OS(conf_df, dataType, dbDir_WFD, runType,
                         timescale_file, timeslots, norm_factor,
                         nside, outName, plot_moll=plot_moll)

    plot_summary_wfd(wfd, conf_df, timescale_file)

    """
    print('wwwwwww', wfd.columns)
    OS_WFDs = wfd['dbName'].unique()
    plot_nsn_versus_two(wfd, xvar='year', xleg='year', logy=False,
                        bins=np.arange(0.5, 11.5, 1), norm_factor=norm_factor,
                        nside=nside,
                        cumul=False, xlim=[1, 10], figtitle=OS_WFDs[0])

    """
    """
    plot_nsn_versus_two(wfd, xvar='z', xleg='z', logy=True,
                        bins=np.arange(0.005, 0.805, 0.01), norm_factor=norm_factor,
                        cumul=True, xlim=[0.01, 0.8])
    """
    """
    for dbName in wfd['dbName'].unique():
        idx = wfd['dbName'] == dbName
        selwfd = wfd[idx]
        mypl = Plot_nsn_vs(selwfd, norm_factor, nside=64)

        mypl.plot_nsn_mollview(what='year', dbName=dbName)

    print(len(wfd))
    """


class Plot_density:
    def __init__(self, data, timescale, dbName='', norm_factor=10, nside=64):
        """
        Class to plot SNe Ia density vs time

        Parameters
        ----------
        data : pandas df
            Data to process.
        timescale : str
            Time scale (year/season).
        dbName : str, optional
            OS name. The default is ''.
        norm_factor : int, optional
            Normalization factor. The default is 10.
        nside : int, optional
            healpix nside parameter. The default is 64.

        Returns
        -------
        None.

        """

        self.data = data
        self.timescale = timescale
        self.dbName = dbName
        self.norm_factor = norm_factor
        self.nside = nside
        import healpy as hp
        self.pixSize = hp.nside2pixarea(self.nside, degrees=True)

        self.plot_hist()
        # self.plot()

    def plot_hist(self):
        """
        Method to plot histos of SNe Ia density (per deg2)

        Returns
        -------
        None.

        """

        nsn = self.data.groupby([self.timescale]).apply(
            lambda x: self.get_nsn(x)).reset_index()

        # fig, ax = plt.subplots(figsize=(14, 9))

        seasons = nsn[self.timescale].unique()

        for seas in seasons:
            fig, ax = plt.subplots()
            fig.suptitle('season {}'.format(seas))
            idx = nsn[self.timescale] == seas
            seldata = nsn[idx]
            ax.hist(seldata['size'], histtype='step', bins=50)

    def plot(self):
        """
        Method to plot

        Returns
        -------
        None.

        """

        fig, ax = plt.subplots(figsize=(14, 9))
        fig.suptitle(r'{}'.format(self.dbName))

        config = ['a', 'b']
        sigma_mus = [1.e6, 0.12]
        sigmas = dict(zip(config, sigma_mus))
        ls = dict(zip(config, ['solid', 'dashed']))

        for key, vals in sigmas.items():
            label = None
            if key == 'b':
                label = '$\sigma_{\mu}\leq$ 0.12'
            nsn_dens = self.get_density(sigma_mu=vals)
            self.plot_density(nsn_dens, what='area',
                              ls=ls[key], label=label, fig=fig, ax=ax)

        ax.set_xlim([1, 10])
        ax.set_xlabel(r'{}'.format(self.timescale), fontweight='bold')
        ax.set_ylabel(r'$N_{SN}/deg^{2}$')
        ax.legend(loc='upper left', bbox_to_anchor=(
            0.1, 1.1), ncol=3, fontsize=15, frameon=False)
        ax.grid(visible=True)

    def get_density(self, sigma_mu=0.12):
        """
        Method to estimate density

        Parameters
        ----------
        sigma_mu : float, optional
            sigma_mu selection criteria. The default is 0.12.

        Returns
        -------
        dens : pandas df
            Density vs time.

        """

        idx = self.data['sigma_mu'] <= sigma_mu
        data = pd.DataFrame(self.data[idx])

        # estimate densities here
        dens = data.groupby([self.timescale]).apply(
            lambda x: self.get_density_season(x)).reset_index()

        return dens

    def get_nsn(self, grp):
        """
        MEthod to estimate the number of SNe IA/pixel/deg2

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        nsn : pandas df
            SNe Ia/deg2/pixel.

        """

        nsn = grp.groupby(['healpixID']).size().to_frame(
            'size').reset_index()
        nsn['size'] /= self.pixSize*self.norm_factor

        return nsn

    def get_density_season(self, grp):
        """
        Method to estimate density for grp

        Parameters
        ----------
        grp : pandas df group
            Data to process.

        Returns
        -------
        pandas df
            SNe Ia density.

        """
        nsn = grp.groupby(['healpixID']).size().to_frame(
            'size').reset_index()

        area = len(nsn)*self.pixSize
        nsn_std = nsn['size'].std()
        nsn_mean = nsn['size'].mean()
        print(nsn_mean, nsn_std)
        idx = np.abs(nsn['size']-nsn_mean) <= nsn_std
        nsn = nsn[idx]
        dens = nsn['size'].median()/self.pixSize
        dens_mean = nsn['size'].sum()/area

        return pd.DataFrame({'density': [dens/self.norm_factor],
                             'density_mean': [dens_mean/self.norm_factor],
                             'area': [area]})

    def plot_density(self, nsn_dens, what='density_mean',
                     ls='solid', label=None, fig=None, ax=None):
        """
        Method to plot density

        Parameters
        ----------
        nsn_dens : pandas df
            Data to plot.
        what: str, optional
           var to plot. The default is density_mean
        ls : str, optional
            Line style. The default is 'solid'.
        label : str, optional
            Plot label. The default is None.
        fig : matplotlib figure, optional
            Figure for the plot. The default is None.
        ax : matplotlib axis, optional
            Axis for the plot. The default is None.

        Returns
        -------
        None.

        """

        if fig is None:
            fig, ax = plt.subplots(figsize=(14, 9))

        ax.plot(nsn_dens[self.timescale], nsn_dens[what],
                color='k', linestyle=ls, label=label)


def plot_summary_wfd(wfd, conf_df, timescale='season', cumul=False):
    """
    Method to plot nsn vs year

    Parameters
    ----------
    wfd : pandas df
        Data to process.
    conf_df : pandas df
        config for plot.
    timescale : str, optional
        Time scale to use (season/year). The default is 'season'.
    cumul : bool, optional
        To plot cumulative results. The default is False.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots(figsize=(13, 8))
    for dbName in wfd['dbName'].unique():
        idx = wfd['dbName'] == dbName
        sel = wfd[idx]
        idc = conf_df['dbName_WFD'] == dbName
        selp = conf_df[idc]
        ls = selp['ls'].values[0]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        plot_versus(sel, fig=fig, ax=ax, cumul=cumul,
                    ls=ls, marker=marker, color=color, mfc=color, label=dbName)
        labelb = dbName+' - '+'$\sigma_{\mu}\leq \sigma_{int}$'
        plot_versus(sel, yvar='nsn_sigma_mu', fig=fig, ax=ax, cumul=cumul,
                    ls='dotted', marker=marker, color=color,
                    mfc='None', label=labelb)

    ax.grid()
    ax.set_xlim([0.95, 10.05])
    ax.set_xlabel(timescale, fontweight='bold')
    legy = '$N_{SN}$'
    if cumul:
        '$\Sigma N_{SN}$'
    ax.set_ylabel(legy)
    # 0, 1.15 for multiple OS
    ax.legend(loc='upper left', bbox_to_anchor=(
        0.1, 1.1), ncol=3, fontsize=15, frameon=False)

    if cumul:
        xmin, xmax = ax.get_xlim()

        nsn = 1.e6
        ax.plot([xmin, xmax], [nsn, nsn],
                color='dimgrey', lw=2, linestyle='solid')
        ax.text(5, 1.02e6, '1 million SNe Ia', color='dimgrey', fontsize=12)
        nsn = 300000
        ax.plot([xmin, xmax], [nsn, nsn],
                color='dimgrey', lw=2, linestyle='solid')
        ax.text(5, 0.32e6, '300k SNe Ia', color='dimgrey', fontsize=12)

    print('here showing')
    plt.show()


def get_stat(grp, norm_factor, sigma_mu=0.12):

    nsn = len(grp)/norm_factor

    idx = grp['sigma_mu'] <= sigma_mu

    nsn_sel = len(grp[idx])/norm_factor

    dictout = dict(zip(['nsn', 'nsn_sigma_mu'], [[nsn], [nsn_sel]]))

    return pd.DataFrame.from_dict(dictout)


def process_DDF(conf_df, dataType, dbDir_DD, runType,
                timescale_file, timeslots,
                norm_factor, nside=128, name='dbName_DD'):

    print('iii', conf_df)
    # load DDF
    OS_DDFs = conf_df[name].unique()

    ddf = pd.DataFrame()
    for OS_DDF in OS_DDFs:
        idx = conf_df[name] == OS_DDF
        fieldType = 'DDF'
        tt = 'load_{}(\'{}\',\'{}\',\'{}\',\'{}\',{},\'{}\')'.format(
            dataType, dbDir_DD, OS_DDF, runType,
            timescale_file, timeslots, fieldType)
        ddfa = eval(tt)
        ddf = pd.concat((ddf, ddfa))

    print('allllllll', timescale_file)
    plot_DDF(ddf, norm_factor, nside=128,
             config=conf_df, timescale=timescale_file)


def process_DDF_summary(conf_df, dataType, dbDir_DD, runType,
                        timescale_file):

    # load DDF
    OS_DDFs = conf_df['dbName_DD'].unique()

    ddf = pd.DataFrame()

    for OS_DDF in OS_DDFs:
        idx = conf_df['dbName_DD'] == OS_DDF
        fieldType = 'DDF'
        search_path = '{}/{}/DDF_{}/nsn_{}_{}.hdf5'.format(
            dbDir_DD, OS_DDF, runType, OS_DDF, timescale_file)
        res = pd.read_hdf(search_path)
        res['dbName'] = OS_DDF
        ddf = pd.concat((ddf, res))

    fields = ddf['field'].unique()

    for field in fields:
        fig, ax = plt.subplots()
        idx = ddf['field'] == field
        sela = ddf[idx]
        dbNames = sela['dbName']
        for dbName in dbNames:
            idxa = sela['dbName'] == dbName
            selb = sela[idxa]
            ax.plot(selb['year'], selb['nsn'])

    plt.show()


parser = OptionParser(description='Script to analyze SN prod after selection')

parser.add_option('--dbDir_DD', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
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
parser.add_option('--timescale_file', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--timeslots', type=str,
                  default='1-10',
                  help='time slot (season or year) to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')
parser.add_option('--plot_Mollweid', type=int,
                  default=0,
                  help='Mollweid view plot for WFD [%default]')


opts, args = parser.parse_args()

dbDir_DD = opts.dbDir_DD
norm_factor_DD = opts.norm_factor_DD
dbDir_WFD = opts.dbDir_WFD
norm_factor_WFD = opts.norm_factor_WFD
budget_DD = opts.budget_DD
runType = opts.runType
config = opts.config
timeslots = opts.timeslots
timescale_file = opts.timescale_file
timeslots = get_val(timeslots)
plot_moll = opts.plot_Mollweid

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
if dbDir_WFD != '':

    process_WFD(conf_df, dataType, dbDir_WFD, runType,
                timescale_file, timeslots, norm_factor_WFD,
                nside=64, plot_moll=plot_moll)

# print(test)

if dbDir_DD != '':

    process_DDF(conf_df, dataType, dbDir_DD, runType,
                timescale_file, timeslots, norm_factor_DD)
    """
    process_DDF_summary(conf_df, dataType, dbDir_DD, runType,
                        timescale_file)
    """
plt.show()
