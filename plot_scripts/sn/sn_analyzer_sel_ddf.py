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


def plot_DDF(data, norm_factor, config, nside=128, timescale='year', timeslots='None'):
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
    from sn_plotter_analysis.sn_analyser_ddf import plot_DDF_nsn
    from sn_plotter_analysis.sn_analyser_ddf import plot_survey_features

    idx = data['zmeas'] >= 0.8
    sel = data[idx]
    sigma_mu = 0.12
    yleg = '$N_{SN} (z\geq 0.8, \sigma_{\mu}\leq\sigma_{int})$'
    plot_DDF_nsn(sel, norm_factor, config, nside,
                 sigma_mu=sigma_mu, timescale=timescale, yleg=yleg)

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

opts, args = parser.parse_args()

dbDir = opts.dbDir
norm_factor = opts.norm_factor
budget_DD = opts.budget_DD
runType = opts.runType
config = opts.config
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
# plot_moll = opts.plot_Mollweid

dataType = opts.dataType

# read config file
conf_df = pd.read_csv(config, comment='#')

# process data
ddf = process_DDF(conf_df, dataType, dbDir, runType,
                  timescale, timeslots, norm_factor)


print('allllllll', timescale)
# plot
plot_DDF(ddf, norm_factor, nside=128, config=conf_df,
         timescale=timescale, timeslots=timeslots)

plt.show()
