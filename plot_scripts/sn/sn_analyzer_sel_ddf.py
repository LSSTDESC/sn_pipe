#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:16:35 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_analysis import plt
import numpy as np

from sn_plotter_analysis.sn_analyser_summary import process_DDF
from sn_plotter_analysis.sn_analyser_ddf import get_val
from sn_plotter_analysis.sn_analyser_ddf import get_nsn
from sn_plotter_analysis.sn_plot import plot_nsn_year_all


def plot_ddf_year(datab, norm_factor, config, nside=128,
                  cols=['year', 'dbName'],
                  fields=['COSMOS', 'CDFS',
                          'XMM-LSS',
                          'ELAISS1', 'EDFS_a', 'EDFS_b']):
    """
    Function to plot nsn vs year

    Parameters
    ----------
    datab : pandas df
        Data to process.
    norm_factor : float
        norm factor.
    config : pandas df
        configuration for the plot.
    nside : int, optional
        nside healpix param. The default is 128.
    cols : list(str), optional
        columns to select data. The default is ['year', 'dbName'].
    fields : list(str), optional
        List of DDFs to consider. The default is 
        ['COSMOS', 'CDFS','XMM-LSS','ELAISS1', 'EDFS_a', 'EDFS_b'].

    Returns
    -------
    None.

    """
    print('aooooo', config)
    # plot nsn vs year
    plot_nsn_tot(datab, norm_factor, config, nside=128,
                 cols=['year', 'dbName'], cumul=False, fields=fields)
    # plot nsn vs year - cumulative
    plot_nsn_tot(datab, norm_factor, config, nside=128,
                 cols=['year', 'dbName'], cumul=True, fields=fields)

    # zmin > 0.8

    plot_nsn_tot(datab, norm_factor, config, nside=128,
                 cols=['year', 'dbName'], cumul=False, fields=fields, zmin=0.8, sigmaC=0.04)
    # plot nsn vs year - cumulative
    plot_nsn_tot(datab, norm_factor, config, nside=128,
                 cols=['year', 'dbName'], cumul=True, fields=fields, zmin=0.8, sigmaC=0.04)

    # plot ratio nsn(z>zmin,sigmac<sigmaC_max)/nsn(z>zmin)
    plot_ratio_sigmac(datab, norm_factor, config, nside=128,
                      cols=['year', 'dbName'],
                      fields=fields, zmin=0.8, sigmaC_max=0.04)


def plot_nsn_tot(datab, norm_factor, config, nside=128,
                 cols=['year', 'dbName'], cumul=False, fields=['COSMOS'],
                 zmin=-1, sigmaC=-1):
    """
    Function to plot nsn (no sel) vs year

    Parameters
    ----------
    datab : pandas df
        Data to process.
    norm_factor : float
        norm factor.
    config : pandas df
        configuration for the plot.
    nside : int, optional
        nside healpix param. The default is 128.
    cols : list(str), optional
        List of cols (groupby) to estimate nsn. The default is ['year', 'dbName'].
    cumul : bool, optional
        To plot cumulative or not. The default is False.
    fields : list(str), optional
        List of DDFs to consider. The default is ['COSMOS'].

    Returns
    -------
    None.

    """

    idx = datab['field'].isin(fields)

    ylab_add = ''
    if zmin > -1:
        idx &= datab['zmeas'] >= zmin
        ylab_add = '$z \geq $'+'{}'.format(zmin)

    if sigmaC > -1:
        idx &= datab['sigmaC'] <= sigmaC
        ylab_add += ', $\sigma_C \leq $'+'{}'.format(sigmaC)

    sel = datab[idx]

    nsn_a = get_nsn(sel, norm_factor, nside, cols=cols)

    ylab = '$\Sigma N_{SN}$'
    if ylab_add != '':
        ylab += '({})'.format(ylab_add)

    plot_nsn_year_all(nsn_a, config,
                      xvar='year', xlab='year',
                      yvar='nsn', ylab=ylab,
                      cumul=cumul, figtit=','.join(fields))


def plot_ratio_sigmac(datab, norm_factor, config, nside=128,
                      cols=['year', 'dbName'],
                      fields=['COSMOS', 'CDFS', 'XMM-LSS',
                              'ELAISS1', 'EDFS_a', 'EDFS_b'],
                      zmin=0.8, sigmaC_max=0.04):
    """
    plot ratio nsn(z>zmin, sigmaC<=sigmaC_max)/nsn(z>zmin)

    Parameters
    ----------
    datab : pandas df
        Data to process.
    norm_factor : float
        norm factor.
    config : pandas df
        configuration for the plot.
    nside : int, optional
        nside healpix param. The default is 128.
    cols : list(str), optional
        List of cols (groupby) to estimate nsn. The default is ['year', 'dbName'].
    fields : list(str), optional
        List of DDFs to consider. 
        The default is ['COSMOS', 'CDFS', 'XMM-LSS','ELAISS1', 'EDFS_a', 'EDFS_b'].
    zmin : float, optional
        Min redshift. The default is 0.8.
    sigmaC_max : float, optional
        Max sigmaC value. The default is 0.04.

    Returns
    -------
    None.

    """

    idm = datab['field'].isin(fields)
    data = datab[idm]
    idx = data['zmeas'] >= zmin
    sel = data[idx]

    nsn_b = get_nsn(sel, norm_factor, nside, cols=cols)

    idx &= data['sigmaC'] <= sigmaC_max
    sel = data[idx]

    nsn_c = get_nsn(sel, norm_factor, nside, cols=cols)

    nsn_rat = nsn_b.merge(nsn_c, left_on=cols, right_on=cols)

    nsn_rat['nsn_ratio'] = nsn_rat['nsn_y']/nsn_rat['nsn_x']

    ylab = '$\\frac{N_{SN}^{z \geq ' + '{}'.format(zmin)
    ylab += ',\sigma_C \leq '+'{}'.format(sigmaC_max)
    ylab += '}}{N_{SN}^{z \geq '+'{}'.format(zmin)+'}}$'
    plot_nsn_year_all(nsn_rat, config,
                      xvar='year', xlab='year',
                      yvar='nsn_ratio', ylab=ylab, cumul=False, figtit=','.join(fields))


def plot_survey_features(data, norm_factor, config, nside=128,
                         timescale='year', timeslots='None', cumul=False):
    """
    function to plot ddf data

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
    cumul: bool.
        To plot cumulative (for NSN for example). The default is False

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
    from sn_plotter_analysis.sn_analyser_ddf import plot_DDF_nsn, plot_nsn_new
    from sn_plotter_analysis.sn_analyser_ddf import plot_survey_features

    """
    plot_ddf_year(data, norm_factor, config, nside,
                  cols=[timescale, 'dbName'])

    """
    """
    print(test)
    plt.show()
    sigma_mu = 0.12
    plot_DDF_nsn(data, norm_factor, config, nside,
                 timescale=timescale, cumul=cumul,
                 plots=['nsn_field_OS'])

    idx = data['zmeas'] >= 0.8
    idx &= data['sigma_mu'] <= sigma_mu
    sel = data[idx]

    yleg_add = '$(z\geq 0.8, \sigma_{\mu}\leq\sigma_{int})$'
    plot_DDF_nsn(sel, norm_factor, config, nside,
                 timescale=timescale, yleg_add=yleg_add,
                 cumul=cumul, plots=['nsn_os', 'pix_area'])

    plt.show()
    # specific survey features
    print(data.columns)
    """
    fields = ['XMM-LSS']
    dbNames = data['dbName'].unique()

    for field in fields:
        for dbName in dbNames:
            plot_survey_features(data, field, dbName, norm_factor, config, nside,
                                 timescale=timescale, timeslots=timeslots)
    plt.show()

    # plot_DDF_dither(data, norm_factor, config, nside)

    # plot_DDF_nsn_z(data, norm_factor, nside)

    """
    mypl.plot_nsn_versus_two(xvar='z', xleg='z', logy=True,
                             cumul=True, xlim=[0.01, 1.1])
    mypl.plot_nsn_mollview()
    """


parser = OptionParser(description='Script to analyze SN - DDF after selection')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--config', type=str,
                  default='input/plots/config_ana.csv',
                  help='OS DD list[%default]')
parser.add_option('--norm_factor', type=int,
                  default=30,
                  help='normalization factor [%default]')
parser.add_option('--budget_DD', type=float,
                  default=0.07,
                  help='DD budget [%default]')
parser.add_option('--runType', type=str,
                  default='spectroz',
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
                  default='nsn_all,nsn_ud',
                  help='for cumulative plots [%default]')
"""
parser.add_option('--cumul', type=int,
                  default=0,
                  help='for cumulative plots [%default]')
"""

opts, args = parser.parse_args()

dbDir = opts.dbDir
norm_factor = opts.norm_factor
budget_DD = opts.budget_DD
runType = opts.runType
config = opts.config
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
plots = opts.plots.split(',')
# cumul = opts.cumul
# plot_moll = opts.plot_Mollweid

dataType = opts.dataType

# read config file
conf_df = pd.read_csv(config, comment='#')

# process data
ddf = process_DDF(conf_df, dataType, dbDir, runType,
                  timescale, timeslots, norm_factor)

print(ddf.columns)
# plot
# all fields
if 'nsn_all' in plots:
    fields = ['COSMOS', 'CDFS', 'XMM-LSS', 'ELAISS1', 'EDFS_a', 'EDFS_b']
    plot_ddf_year(ddf, norm_factor, conf_df, nside=128,
                  cols=['year', 'dbName'],
                  fields=fields)
if 'nsn_ud' in plots:
    # UD only

    fields = ['COSMOS', 'XMM-LSS']
    plot_ddf_year(ddf, norm_factor, conf_df, nside=128,
                  cols=['year', 'dbName'],
                  fields=fields)


plt.show()

plot_survey_features(ddf, norm_factor, nside=128, config=conf_df,
                     timescale=timescale, timeslots=timeslots, cumul=cumul)

plt.show()
