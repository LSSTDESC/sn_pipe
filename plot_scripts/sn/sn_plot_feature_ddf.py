#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:16:56 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import matplotlib.pyplot as plt
from sn_plotter_analysis.sn_analyser_summary import load_data
from sn_plotter_analysis.sn_analyser_ddf import get_val
from sn_plotter_analysis.sn_analyser_ddf import plot_survey_features
from optparse import OptionParser


def plot_survey_feat(data, norm_factor, nside=128,
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
    # from sn_plotter_analysis.sn_analyser_ddf import plot_DDF_nsn, plot_nsn_new

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
    # fields = ['XMM-LSS']
    dbNames = data['dbName'].unique()

    for field in fields:
        for dbName in dbNames:
            plot_survey_features(data, field, dbName, norm_factor, nside,
                                 timescale=timescale, timeslots=timeslots)
    plt.show()

    # plot_DDF_dither(data, norm_factor, config, nside)

    # plot_DDF_nsn_z(data, norm_factor, nside)

    """
    mypl.plot_nsn_versus_two(xvar='z', xleg='z', logy=True,
                             cumul=True, xlim=[0.01, 1.1])
    mypl.plot_nsn_mollview()
    """


parser = OptionParser(description='Script to plot DDF features')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS to process[%default]')
parser.add_option('--norm_factor', type=int,
                  default=30,
                  help='normalization factor [%default]')
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
parser.add_option('--fields', type=str,
                  default='COSMOS,XMM-LSS,CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='data type [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
norm_factor = opts.norm_factor
runType = opts.runType
timeslots = get_val(opts.timeslots)
timescale = opts.timescale
dataType = opts.dataType
fields = opts.fields.split(',')

fieldType = 'DDF'

ddf = load_data(dbDir, dbName, dataType, runType,
                timescale, timeslots, fieldType)

plot_survey_feat(ddf, norm_factor, nside=128,
                 timescale=timescale, timeslots=timeslots)
