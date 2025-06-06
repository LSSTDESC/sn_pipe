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
import os

from sn_plotter_analysis.sn_analyser_summary import process_DDF
from sn_plotter_analysis.sn_analyser_ddf import get_val
from sn_plotter_analysis.sn_analyser_ddf import get_nsn, pixelSize
from sn_plotter_analysis.sn_plot import plot_nsn_year_all
from sn_plotter_analysis.sn_analyser_tools import Estimate_NSN, count_all
from sn_plotter_analysis.sn_analyser_tools import clean_level, print_nsn_latex
from sn_tools.sn_io import checkDir


def plot_nsn_tot(nsn_a, config,
                 cumul=False,
                 yvar='nsn', ylab='$\Sigma N_{SN}$',
                 yvar_err='', fields=['COSMOS']):
    """
    Function to plot nsn (no sel) vs year

    Parameters
    ----------
    nsn_a : pandas df
        Data to process.
    config : pandas df
        configuration for the plot.

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

    plot_nsn_year_all(nsn_a, config,
                      xvar='year', xlab='year',
                      yvar=yvar, ylab=ylab, yvar_err=yvar_err,
                      cumul=cumul, figtit=','.join(fields))


def plot_ddf_year(data, config,
                  cols=['year', 'dbName'],
                  fields=['COSMOS', 'CDFS',
                          'XMM-LSS',
                          'ELAISS1', 'EDFS_a', 'EDFS_b']):
    """
    Function to plot nsn vs year

    Parameters
    ----------
    data : pandas df
        Data to process.
    config : pandas df
        configuration for the plot.
    cols : list(str), optional
        columns to select data. The default is ['year', 'dbName']
    fields : list(str), optional
        List of DDFs to consider. The default is
        ['COSMOS', 'CDFS','XMM-LSS','ELAISS1', 'EDFS_a', 'EDFS_b'].

    Returns
    -------
    None.

    """

    idx = data['field'].isin(fields)
    data = pd.DataFrame(data[idx])
    # plot nsn vs year - with stat error

    datab = count_all(
        data, cols, var=['nsn', 'survey_area'], err_var=['err_nsn'])

    datab['nsn_sqdeg'] = datab['nsn']/datab['survey_area']
    datab['err_nsn_sqdeg'] = datab['err_nsn']/datab['survey_area']
    for cumul in [False, True]:
        plot_nsn_tot(datab, config, yvar='nsn', yvar_err='err_nsn',
                     cumul=cumul, fields=fields)

    ylab = '$N_{SN}/deg^2$'
    plot_nsn_tot(datab, config, yvar='nsn_sqdeg',
                 yvar_err='err_nsn_sqdeg', ylab=ylab,
                 cumul=False, fields=fields)

    # zmin > 0.8, sigmac<=0.04
    var = ['nsn_z_08_sigmaC', 'survey_area']
    err_var = 'err_nsn_z_08_sigmaC'
    datab = count_all(data, cols, var=var, err_var=[err_var])
    datab = clean_level(datab)

    ylab_add = '$z \geq $'+'{}'.format(0.8)
    ylab_add += ', $\sigma_C \leq $'+'{}'.format(0.04)
    ylab = '$\Sigma N_{SN}$'
    if ylab_add != '':
        ylab += '({})'.format(ylab_add)

    for cumul in [False, True]:
        plot_nsn_tot(datab, config, yvar=var[0],
                     yvar_err=err_var, ylab=ylab,
                     cumul=cumul, fields=fields)

    datab['nsn_sqdeg'] = datab['nsn_z_08_sigmaC']/datab['survey_area']
    datab['err_nsn_sqdeg'] = datab['err_nsn_z_08_sigmaC']/datab['survey_area']
    ylab = '$N_{SN}/deg^2$'+'({})'.format(ylab_add)
    plot_nsn_tot(datab, config, yvar='nsn_sqdeg',
                 yvar_err='err_nsn_sqdeg', ylab=ylab,
                 cumul=False, fields=fields)

    var = 'nsn_z_08'
    err_var = 'err_nsn_z_08'
    datac = count_all(data, cols, var=[var], err_var=[err_var])
    datac = clean_level(datac)
    datab = datab.drop(columns=['survey_area'])

    data_m = datac.merge(datab, left_on=['dbName', 'year'], right_on=[
                         'dbName', 'year'], suffixes=['', ''])

    data_m['ratio'] = data_m['nsn_z_08_sigmaC']/data_m['nsn_z_08']
    vv = data_m['err_nsn_z_08_sigmaC']**2
    vv /= data_m['nsn_z_08']**2
    vvb = (data_m['nsn_z_08_sigmaC']/data_m['nsn_z_08']**2)**2
    vvb *= data_m['err_nsn_z_08']**2
    data_m['err_ratio'] = np.sqrt(vvb+vv)

    ylab = '$\\frac{N_{SN}^{z \geq ' + '{}'.format(0.8)
    ylab += ',\sigma_C \leq '+'{}'.format(0.04)
    ylab += '}}{N_{SN}^{z \geq '+'{}'.format(0.8)+'}}$'

    plot_nsn_tot(data_m, config, yvar='ratio',
                 yvar_err='err_ratio', ylab=ylab,
                 cumul=False, fields=fields)


def get_nsn(data, norm_factor, nside=128):
    """
    Function to estimate the number of SNe Ia + errors

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    norm_factor : TYPE
        DESCRIPTION.

    Returns
    -------
    res_fi : TYPE
        DESCRIPTION.

    """

    nsn = Estimate_NSN(norm_factor=norm_factor)

    # get nsn - no cuts
    resa = nsn(data)
    resa = clean_level(resa)

    # get nsn - z >= 0.8
    idx = data['z'] >= 0.8
    sel = data[idx]
    resb = nsn(sel)
    resb = clean_level(resb)
    resb = resb.rename(columns={'nsn': 'nsn_z_08', 'err_nsn': 'err_nsn_z_08'})

    # get nsn - z >= 0.8 and sigmaC <= 0.04
    idx = data['z'] >= 0.8
    idx &= data['sigmaC'] <= 0.04
    sel = data[idx]
    resc = nsn(sel)
    resc = clean_level(resc)
    resc = resc.rename(
        columns={'nsn': 'nsn_z_08_sigmaC', 'err_nsn': 'err_nsn_z_08_sigmaC'})

    cols = ['dbName', 'field', 'healpixID', 'season', 'year']
    res_fi = resa.merge(resb, left_on=cols, right_on=cols, suffixes=['', ''])

    res_fi = res_fi.merge(resc, left_on=cols, right_on=cols, suffixes=['', ''])

    res_fi['survey_area'] = pixelSize(nside)

    return res_fi


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
                  help='plots to draw [%default]')
parser.add_option('--print_nsn', type=int,
                  default=0,
                  help='to print nsn as a latex table [%default]')
parser.add_option('--ud_fields', type=str,
                  default='COSMOS,XMM-LSS',
                  help='UD fields to consider [%default]')
parser.add_option('--dd_fields', type=str,
                  default='CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='DD fields to consider [%default]')
parser.add_option('--nside', type=int,
                  default=128,
                  help='nside healpix parameter [%default]')
parser.add_option('--inputDir', type=str,
                  default='../sn_summary_ddf',
                  help='input dir for the file to draw [%default]')
parser.add_option('--fileName', type=str,
                  default='sn_summary_ddf.hdf5',
                  help='sn file name to draw [%default]')


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
print_nsn = opts.print_nsn
ud_fields = opts.ud_fields.split(',')
dd_fields = opts.dd_fields.split(',')
nside = opts.nside
inputDir = opts.inputDir
fileName = opts.fileName
dataType = opts.dataType


# create dir if necessary
checkDir(inputDir)


# read config file
conf_df = pd.read_csv(config, comment='#')

fName = '{}/{}'.format(inputDir, fileName)

# process data
if not os.path.isfile(fName):
    print('File not found! Processing data')
    ddf = process_DDF(conf_df, dataType, dbDir, runType,
                      timescale, timeslots, norm_factor)

    df_nsn = get_nsn(ddf, norm_factor, nside)

    df_nsn.to_hdf(fName, key='ddf')


df_nsn = pd.read_hdf(fName)

if print_nsn:
    print_nsn_latex(df_nsn)


# all fields
if 'nsn_all' in plots:
    fields = ud_fields+dd_fields
    plot_ddf_year(df_nsn, conf_df,
                  cols=['year', 'dbName'],
                  fields=fields)


if 'nsn_ud' in plots:
    # UD only

    fields = ud_fields
    plot_ddf_year(df_nsn, conf_df,
                  cols=['year', 'dbName'],
                  fields=fields)

plt.show()
