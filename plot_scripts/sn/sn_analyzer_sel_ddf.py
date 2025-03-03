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


def plot_DDF(data, norm_factor, config, nside=128,
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
    plot_nsn_new(data, norm_factor, config, nside,
                 sigma_mu=0.12, timescale=timescale)

    idx = data['zmeas'] >= 0.8
    sel = data[idx]

    plot_nsn_new(sel, norm_factor, config, nside,
                 sigma_mu=0.12, timescale=timescale)

    idx &= data['sigma_mu'] <= 0.12
    sel = data[idx]

    plot_nsn_new(sel, norm_factor, config, nside,
                 sigma_mu=0.12, timescale=timescale)
    print(test)
    """
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
    fields = ['COSMOS']
    dbNames = data['dbName'].unique()

    for field in fields:
        for dbName in dbNames:
            plot_survey_features(data, field, dbName, norm_factor, config, nside,
                                 timescale=timescale, timeslots=timeslots)

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
parser.add_option('--cumul', type=int,
                  default=0,
                  help='for cumulative plots [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
norm_factor = opts.norm_factor
budget_DD = opts.budget_DD
runType = opts.runType
config = opts.config
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
cumul = opts.cumul
# plot_moll = opts.plot_Mollweid

dataType = opts.dataType

# read config file
conf_df = pd.read_csv(config, comment='#')

# process data
ddf = process_DDF(conf_df, dataType, dbDir, runType,
                  timescale, timeslots, norm_factor)

print(ddf.columns)
# plot
plot_DDF(ddf, norm_factor, nside=128, config=conf_df,
         timescale=timescale, timeslots=timeslots, cumul=cumul)

plt.show()
