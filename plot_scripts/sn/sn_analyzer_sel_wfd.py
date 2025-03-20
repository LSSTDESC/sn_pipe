#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:50:54 2023

@author: philippe.gris@clermont.in2p3.fr
"""


# import glob
from optparse import OptionParser
import pandas as pd
import numpy as np
from sn_analysis import plt
from sn_analysis.sn_calc_plot import bin_it, bin_it_mean, bin_it_effi
# import h5py
# from astropy.table import Table
# from sn_tools.sn_utils import multiproc
from sn_plotter_analysis.sn_analyser_summary import process_WFD_OS_nsn
from sn_plotter_analysis.sn_analyser_tools import get_stat
from sn_tools.sn_io import checkDir
import os


def plot_DDF_deprecated(data, norm_factor, config, nside=128, timescale='year'):
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
    idx = data['zmeas'] >= 0.8
    sel = data[idx]
    sigma_mu = 0.12
    yleg = '$N_{SN} (z\geq 0.8, \sigma_{\mu}\leq\sigma_{int})$'
    plot_DDF_nsn(sel, norm_factor, config, nside,
                 sigma_mu=sigma_mu, timescale=timescale, yleg=yleg)

    plot_survey_features(data, norm_factor, config, nside, timescale=timescale)

    # plot_DDF_dither(data, norm_factor, config, nside)

    # plot_DDF_nsn_z(data, norm_factor, nside)

    """
    mypl.plot_nsn_versus_two(xvar='z', xleg='z', logy=True,
                             cumul=True, xlim=[0.01, 1.1])
    mypl.plot_nsn_mollview()
    """


def plot_survey_features_deprecated(data, norm_factor, config, nside, timescale):
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
    dbName = 'baseline_v3.4_10yrs'
    dbName = 'roll_uniform_early_half_mjdp67_v3.4_10yrs'
    # dbName = 'DDF_DESC_0.80_WZ_0.07'

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


def plot_sn_features_deprecated(data, field, dbName, timescale,
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


def plot_sigma_mu_deprecated(ax, selb, xvar, yvar, yvar_cut,
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


def plot_nsn_deprecated(ax, selb, xvar, yvar, yvar_cut,
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


def plot_effi_deprecated(ax, selb, xvar, yvar, yvar_cut,
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


def get_zmax_field_deprecated(data, field, dbName, timescale, zmin, sigmaC):

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


def plot_DDF_nsn_z_deprecated(data, norm_factor, nside, timescale='year'):
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


def plot_DDF_dither_deprecated(data, norm_factor, config, nside, timescale='year'):
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


def plot_DDF_nsn_deprecated(data, norm_factor, config, nside, sigma_mu=1.e6,
                            timescale='year', yleg=''):
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
    sigma_mu: float, optional.
     sigma_mu selection cut. The default is 1.e6
    timescale: str, opt
    Time scale for the plot. The default is 'year'

    Returns
    -------
    None.

    """

    idx = data['sigma_mu'] <= sigma_mu

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

    # plot_field(sums, mypl, config, xvar=timescale,
    #           xleg=timescale, cumul=True)
    # plot_field(sums, mypl, xvar=timescale, xleg=timescale,
    #           yvar='pixArea', yleg='Observed Area [deg$^{2}$]')

    # total number of SN per season/OS
    plot_field(sumt, mypl, config, xvar=timescale, xleg=timescale,
               cumul=True, yleg=yleg)

    # plot_field(sumt, mypl, xvar=timescale, xleg=timescale,
    #           yvar='pixArea', yleg='Observed Area [deg$^{2}$]')
    plt.show()


def get_sums_nsn_deprecated(data, norm_factor, nside, cols=['season', 'dbName', 'field']):
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


def plot_field_deprecated(data, mypl, config, xvar='season', xleg='season',
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


def plot_field_time_deprecated(data, mypl, config, xvar='dist', xleg='dist',
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


class Plot_nsn_vs_deprecated:
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


def plot_nsn_versus_two_deprecated(data, norm_factor=30, nside=128,
                                   bins=np.arange(0.005, 0.81, 0.01),
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
    idx = data['sigma_c'] <= 0.04
    labelb = label+' - $\sigma_C \leq 0.04$'
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

    print('plotting here', xvar, yvar)
    ax.plot(df[xvar], ypl, ls=ls, marker=marker,
            color=color, label=label, mfc=mfc, markersize=9, lw=2)


def plot_nsn_binned_old(data, norm_factor=30, nside=128,
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

    print('bin it', res)
    nsn_tot = np.sum(res['NSN'])
    print('total number of SN', len(data), nsn_tot)

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


def plotMollview(data, varName, figtit, xmin, xmax,
                 nside=128, outDir='.', saveName=''):
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

    hpxmap = np.zeros(npix, dtype=float)
    hpxmap = np.full(hpxmap.shape, 0.)
    hpxmap[data['healpixID'].astype(
        int)] += data[varName]

    print(np.where(hpxmap < 0.01))

    norm = plt.cm.colors.Normalize(xmin, xmax)
    cmap = plt.cm.jet
    cmap.set_under('w')

    hp.mollview(hpxmap, fig=fig, min=xmin, max=xmax, cmap=cmap,
                title=figtit, nest=True, norm=norm)
    hp.graticule()

    if saveName != '':
        plt.savefig('{}/{}'.format(outDir, saveName))

    plt.close()


def plotMollview_deprecated(data, varName, leg, addleg, op, xmin, xmax,
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

    hpxmap = np.zeros(npix, dtype=float)
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


def process_WFD(conf_df, dataType, dbDir_WFD, runType,
                timescale_file, timeslots, norm_factor, fName):
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

    Returns
    -------
    wfd : pandas df
        Output data.

    """

    OS_WFDs = conf_df['dbName_WFD'].unique()
    print('dbNames', OS_WFDs)

    # fig, ax = plt.subplots(figsize=(14, 8))
    from_to_load = 'from sn_plotter_analysis.sn_analyser_tools'
    mod_to_load = '{} import load_{}'.format(from_to_load, dataType)
    exec(mod_to_load)
    for OS_WFD in OS_WFDs:
        idx = conf_df['dbName_WFD'] == OS_WFD
        tt = 'load_{}(\'{}\',\'{}\',\'{}\',\'{}\',{},norm_factor={})'.format(
            dataType, dbDir_WFD, OS_WFD, runType,
            timescale_file, timeslots, norm_factor)
        wfda = eval(tt)
        wfda['dbName'] = OS_WFD
        wfda.to_hdf(fName, key='nsn_WFD')
        del wfda


def process_WFD_OS_deprecated(conf_df, dataType, dbDir_WFD, runType,
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
    # fig, ax = plt.subplots(figsize=(14, 8))
    fig, ax = None, None
    from_to_load = 'from sn_plotter_analysis.sn_analyser_tools'
    mod_to_load = '{} import load_{}'.format(from_to_load, dataType)
    exec(mod_to_load)
    for OS_WFD in OS_WFDs:
        idx = conf_df['dbName_WFD'] == OS_WFD
        tt = 'load_{}(\'{}\',\'{}\',\'{}\',\'{}\',{})'.format(
            dataType, dbDir_WFD, OS_WFD, runType,
            timescale_file, timeslots)
        wfda = eval(tt)
        idx = wfda['fitstatus'] == 'fitok'
        idx &= wfda['ebvofMW'] < 0.25
        # idx &= wfda['sigma_c'] <= 0.04
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

        # plot nsn vs z
        """
        plot_nsn_versus_two(wfda, xvar='z', xleg='z', logy=False,
                            bins=np.arange(0.005, 0.81, 0.01),
                            norm_factor=norm_factor,
                            cumul=True, xlim=[0.01, 0.81], color=color,
                            marker=marker, fig=fig, ax=ax, cumnorm=False)
        """
        # density plot here
        # Plot_density(wfda, timescale_file, OS_WFD)

        # wfd = pd.concat((wfd, wfda))
        # get some stat

        wfda = wfda.groupby(['dbName', timescale_file]).apply(
            lambda x: get_stat(x, norm_factor)).reset_index()
        wfd = pd.concat((wfd, wfda))
        del wfda

    if ax is not None:
        ax.legend(loc='upper left', bbox_to_anchor=(
            0.0, 1.15), ncol=3, fontsize=11, frameon=False)
    plt.show()

    # print(wfd)

    # wfd.to_csv(outName, index=False)

    print('out')
    return wfd


def process_WFD_deprecated(conf, dataType, dbDir, runType,
                           timescale_file, timeslots, norm_factor,
                           nside=64, timescale='season', plot_moll=False):
    """
    Function to process WFD data

    Parameters
    ----------
    conf : pandas df
        Confituration file.
    dataType : str
        Data type.
    dbDir : str
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

    outName = 'res_nsn_wfd_3_6_to.csv'
    """
    outName = 'nsn_WFD_v3_6.csv'
    outName = 'res_nsn_wfd_3_6_new.csv'

    wfd = process_WFD_OS_nsn(conf_df, dataType, dbDir_WFD, runType,
                             timescale_file, timeslots, norm_factor,
                             outName=outName)

    plot_summary_wfd(wfd, conf_df, timescale_file,
                     cumul=True, rem_from_name='v3.0')

    """

    wfd = process_WFD_OS(conf, dataType, dbDir, runType,
                         timescale_file, timeslots, norm_factor,
                         nside)

    plot_summary_wfd(wfd, conf, timescale_file,
                     cumul=True, rem_from_name='v3.0')

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


class Plot_density_deprecated:
    def __init__(self, data, timescale, dbName='', norm_factor=10, nside=64):
        """
        Class to plot SNe Ia density vs time

        Parameters
        ----------
        data: pandas df
            Data to process.
        timescale: str
            Time scale(year/season).
        dbName: str, optional
            OS name. The default is ''.
        norm_factor: int, optional
            Normalization factor. The default is 10.
        nside: int, optional
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
        Method to plot histos of SNe Ia density(per deg2)

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
        sigma_mu: float, optional
            sigma_mu selection criteria. The default is 0.12.

        Returns
        -------
        dens: pandas df
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
        grp: pandas df
            Data to process.

        Returns
        -------
        nsn: pandas df
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
        grp: pandas df group
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
        nsn_dens: pandas df
            Data to plot.
        what: str, optional
           var to plot. The default is density_mean
        ls: str, optional
            Line style. The default is 'solid'.
        label: str, optional
            Plot label. The default is None.
        fig: matplotlib figure, optional
            Figure for the plot. The default is None.
        ax: matplotlib axis, optional
            Axis for the plot. The default is None.

        Returns
        -------
        None.

        """

        if fig is None:
            fig, ax = plt.subplots(figsize=(14, 9))

        ax.plot(nsn_dens[self.timescale], nsn_dens[what],
                color='k', linestyle=ls, label=label)


def plot_summary_wfd(wfda, conf_df, timescale='season',
                     cumul=False):
    """
    Method to plot nsn vs year

    Parameters
    ----------
    wfd: pandas df
        Data to process.
    conf_df: pandas df
        config for plot.
    timescale: str, optional
        Time scale to use(season/year). The default is 'season'.
    cumul: bool, optional
        To plot cumulative results. The default is False.

    Returns
    -------
    None.

    """

    wfd = wfda.groupby(['dbName', timescale])[
        'nsn', 'nsn_cosmo'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.subplots_adjust(right=0.75)
    for dbName in wfd['dbName'].unique():
        idx = wfd['dbName'] == dbName
        sel = wfd[idx]
        idc = conf_df['dbName_WFD'] == dbName
        selp = conf_df[idc]
        ls = selp['ls'].values[0]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        dbNameb = dbName
        plot_versus(sel, fig=fig, ax=ax, cumul=cumul,
                    ls=ls, marker=marker, color=color, mfc=color, label=dbNameb)
        labelb = dbNameb+' - '+'$\sigma_{\mu}\leq \sigma_{int}$'
        labelb = dbNameb+' - '+'$\sigma_C \leq 0.04$'
        labelb = dbNameb+' - '+'cosmo'
        plot_versus(sel, yvar='nsn_cosmo', fig=fig, ax=ax, cumul=cumul,
                    ls='dotted', marker=marker, color=color,
                    mfc='None', label='')

    ax.grid()
    ax.set_xlim([0.95, 10.05])
    ax.set_xlabel(timescale, fontweight='bold')
    legy = '$N_{SN}$'
    if cumul:
        '$\Sigma N_{SN}$'
    ax.set_ylabel(legy)
    # 0, 1.15 for multiple OS
    # ax.legend(loc='upper left', bbox_to_anchor=(
    #    0.1, 1.1), ncol=3, fontsize=15, frameon=False)
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.20, 0.7),
              ncol=1, fontsize=12, frameon=False)
    if cumul:
        xmin, xmax = ax.get_xlim()

        nsn = 1.e6
        ax.plot([xmin, xmax], [nsn, nsn],
                color='dimgrey', lw=2, linestyle='solid')
        ax.text(5, 1.02e6, '1 million SNe Ia', color='dimgrey', fontsize=12)
        nsn = 200000
        ax.plot([xmin, xmax], [nsn, nsn],
                color='dimgrey', lw=2, linestyle='solid')
        ax.text(5, 0.22e6, '200k SNe Ia', color='dimgrey', fontsize=12)

    print('here showing')
    plt.show()


def plot_mollview_wfd(data, timescale, timeslots, nside, varp='nsn', outDir='.'):
    """
    Function to make Mollweid plots for nsn in the WFD survey

    Parameters
    ----------
    data : pandas df
        Data to process.
    timescale : str
        Time scale (year/season).
    timeslots : list
        List of season/years to plot.
    nside : int
        healpix nside parameter.
    varp : str, optional
        var to plot. The default is 'nsn'.
    outDir : str, optional
        output directory to save the plot. The default is '.'.

    Returns
    -------
    None.

    """

    dbNames = data['dbName'].unique()

    varleg = 'N$_{SN}$='
    for dbName in dbNames:
        idx = data['dbName'] == dbName
        sel = data[idx]
        # plot all seasons
        xmin = sel[varp].min()
        xmax = sel[varp].max()
        nsn = int(np.sum(sel[varp]))
        figtit = '{} \n'.format(dbName)
        figtitb = figtit + varleg
        figtitb += '{}'.format(nsn)
        outDirName = '{}/{}'.format(outDir, dbName)
        checkDir(outDirName)
        saveName = 'nsn.png'
        plotMollview(sel, varp, figtitb, xmin, xmax, nside=nside,
                     outDir=outDirName, saveName=saveName)
        # season by season
        for timesl in timeslots:
            idxb = sel[timescale] == timesl
            selb = sel[idxb]
            xmin = selb[varp].min()
            xmax = selb[varp].max()
            nsn = int(np.sum(selb[varp]))
            figtitb = figtit + '{} {} '.format(timescale, timesl)
            figtitb += varleg+'{}'.format(nsn)
            saveName = 'nsn_{}_{}.png'.format(timescale, timesl)
            plotMollview(selb, varp, figtitb,
                         xmin, xmax, nside=nside,
                         outDir=outDirName, saveName=saveName)


def plot_density_wfd(datam, timescale, timeslots, nside, conf_df,
                     varp='nsn', norm_factor=10, plot_indiv=False):
    """
    Function to plot SN densities

    Parameters
    ----------
    datam : pandas df
        Data to process.
    timescale : str
        Timescale to use.
    timeslots : list(int)
        Time slots to select.
    nside : int
        nside healpix parameter.
    conf_df : pandas df
        Config for the plot.
    varp : str, optional
        Data to consider. The default is 'nsn'.
    norm_factor : float, optional
        WFD norm factor. The default is 10.
    plot_indiv : bool, optional
        to plot indiv. The default is False.

    Returns
    -------
    None.

    """

    print(datam.columns)

    idx = datam[varp] > 0.
    data = datam[idx]
    data[varp] /= norm_factor

    data['healpixID'] = data['healpixID'].astype(int)
    healpixId = data['healpixID'].unique().tolist()
    df_pix = pix_RA_Dec(healpixId, nside)
    data = data.merge(df_pix, left_on=['healpixID'], right_on=[
        'healpixID'], suffixes=['', ''])

    dbNames = data['dbName'].unique()
    ylabel = 'N$_{SN}$/deg$^{2}$'
    if varp == 'nsn_cosmo':
        ylabel = 'N$_{SN}^{cosmo}$/deg$^{2}$'
    fig, ax = plt.subplots(figsize=(12, 8))
    figb, axb = plt.subplots(figsize=(12, 8))
    for dbName in dbNames:
        idx = data['dbName'] == dbName
        sel = data[idx]
        idc = conf_df['dbName_WFD'] == dbName
        selp = conf_df[idc]
        ls = selp['ls'].values[0]
        marker = selp['marker'].values[0]
        color = selp['color'].values[0]
        vara = '{}_density_mean'.format(varp)
        varb = '{}_density_std'.format(varp)
        dfa = sel.groupby(['healpixID', 'pixRA', 'pixDec'])[
            varp].sum().reset_index()
        df = get_nsn_dec(dfa, varp, delta_dec=5.)
        if plot_indiv:
            plot_density_os_summary(df, vara, varb,
                                    fig=None, ax=None,
                                    ylabel=ylabel, figtit=dbName,
                                    ls=ls, color=color,
                                    marker=marker, label='')
        plot_density_os_summary(df, vara, '', fig=fig, ax=ax,
                                ylabel=ylabel, figtit=dbName,
                                ls=ls, color=color, marker=marker, label='')
        plot_density_os_summary(df, '{}_area'.format(varp), '',
                                fig=figb, ax=axb, ylabel='area [deg$^2$]',
                                figtit=dbName, ls=ls, color=color,
                                marker=marker, label='')

    ax.grid(visible=True)
    ax.set_xlabel(r'Dec [deg]')
    ax.set_ylabel(r'{}'.format(ylabel))
    axb.grid(visible=True)
    axb.set_xlabel(r'Dec [deg]')
    axb.set_ylabel(r'area [deg$^2$]')
    plt.show()


def plot_density_os_summary(df, varm, varstd, fig=None, ax=None,
                            ylabel='N$_{SN}$/deg$^{2}$',
                            figtit='', ls='None', color='k', marker='o',
                            label=''):
    """
    Function to make the plot

    Parameters
    ----------
    df : pandas df
        Data to plot.
    varm : str
        y-axis var mean.
    varstd : str
        y-axis vzr std.
    fig : matplotlib figure, optional
        Figure for the plot. The default is None.
    ax : matplotlib axis, optional
        axis for the plot. The default is None.
    ylabel : str, optional
        y-label. The default is 'N$_{SN}$/deg$^{2}$'.
    figtit : str, optional
        Figure title. The default is ''.
    ls : str, optional
        Line style. The default is 'None'.
    color : color, optional
        color for the plot. The default is 'k'.
    marker : str, optional
        marker for the plot. The default is 'o'.
    label : str, optional
        label for legend. The default is ''.

    Returns
    -------
    None.

    """

    draw_indiv = False
    if fig is None:
        draw_indiv = True
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(figtit)

    ax.plot(df['dec'], df['{}'.format(varm)], color=color,
            linestyle=ls, marker=marker, mfc='None', label=label)
    # ax.errorbar(df['dec'], df['nsn_density_mean'],
    #            yerr=df['nsn_density_std'], color='k')
    if draw_indiv:
        df['plus'] = df['{}'.format(varm)]+df['{}'.format(varstd)]
        df['minus'] = df['{}'.format(varm)]-df['{}'.format(varstd)]
        ax.fill_between(df['dec'], df['plus'],
                        df['minus'], color='yellow')

        ax.grid(visible=True)
        ax.set_xlabel(r'Dec [deg]')
        ax.set_ylabel(r'{}'.format(ylabel))


def pix_RA_Dec(healpixId, nside):
    """
    Function to grab (pixRA,pixDec) from healpixID list

    Parameters
    ----------
    healpixId : list(int)
        List of healpixIDs.
    nside : int
        nside healpix parameter.

    Returns
    -------
    data : pandas df
        results: df['healpixID','pixRA','pixDec'].

    """

    import healpy as hp
    coord = hp.pix2ang(nside, healpixId, nest=True, lonlat=True)
    df_pix = pd.DataFrame(healpixId, columns=['healpixID'])

    df_pix['pixRA'] = coord[0]
    df_pix['pixDec'] = coord[1]

    return df_pix


def get_nsn_dec(data, varp='nsn', delta_dec=5.):
    """
    Function to estimate the varp density and area per Dec slices

    Parameters
    ----------
    data : pandas df
        Data to process.
    varp : str, optional
        var to consider. The default is 'nsn'.
    delta_dec : float, optional
        dec slice width. The default is 5..

    Returns
    -------
    df : pandas df
        output data.

    """

    decs = np.arange(-80., 20., delta_dec)
    bin_centers = (decs[: -1] + decs[1:])/2
    df = pd.DataFrame(bin_centers, columns=['dec'])
    df['dec'] -= delta_dec/2.

    group = data.groupby(pd.cut(data['pixDec'], decs))

    pixSize = pixelSize(nside)
    df[f'{varp}_sum'] = group[varp].sum().to_list()
    df[f'{varp}_density_mean'] = group[varp].mean().to_list()
    df[f'{varp}_density_std'] = group[varp].std().to_list()
    df[f'{varp}_density_mean'] /= pixSize
    df[f'{varp}_density_std'] /= pixSize
    df[f'{varp}_area'] = group.size().to_list()
    df[f'{varp}_area'] *= pixSize

    return df


parser = OptionParser(description='Script to analyze SN prod after selection')

parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS list[%default]')
parser.add_option('--dbDir', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--norm_factor', type=int,
                  default=10,
                  help='Normalization factor [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz_nosat',
                  help='run type  [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--timeslots', type=str,
                  default='1-10',
                  help='time slot (season or year) to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')
parser.add_option('--plots', type=str,
                  default='summary,mollweid,density',
                  help='plots to draw [%default]')
parser.add_option('--outDir', type=str,
                  default='../sn_wfd',
                  help='output dir [%default]')
parser.add_option('--outName', type=str,
                  default='nsn_wfd.hdf5',
                  help='output name [%default]')
parser.add_option('--nside', type=int,
                  default=64,
                  help='healpix nside parameter [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
norm_factor = opts.norm_factor
runType = opts.runType
config = opts.config
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
dataType = opts.dataType
plots = opts.plots.split(',')
outDir = opts.outDir
outName = opts.outName
nside = opts.nside

# read config file
conf = pd.read_csv(config, comment='#')

# check outputdir
checkDir(outDir)

dbNames = conf['dbName_WFD'].unique()

wfd = pd.DataFrame()
for dbName in dbNames:
    print('loading', dbName)
    # check outputdir
    fName = f'{outDir}/{dbName}/{outName}'
    if not os.path.isfile(fName):
        print('file not found', fName)
        # load wfds
        checkDir(f'{outDir}/{dbName}')
        idx = conf['dbName_WFD'] == dbName
        process_WFD(conf[idx], dataType, dbDir, runType,
                    timescale, timeslots, norm_factor, fName)
    wfda = pd.read_hdf(fName)
    wfd = pd.concat((wfd, wfda))


print(wfd['dbName'].unique())
if 'summary' in plots:
    print('timescel', timescale)
    plot_summary_wfd(wfd, conf, timescale, cumul=True)

if 'mollweid' in plots:
    plot_mollview_wfd(wfd, timescale, timeslots, nside, outDir=outDir)

if 'density' in plots:
    plot_density_wfd(wfd, timescale, timeslots, nside, conf,
                     varp='nsn', plot_indiv=True)

plt.show()
