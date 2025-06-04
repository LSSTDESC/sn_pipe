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


class Estimate_NSN:
    def __init__(self, norm_factor=30,
                 rate='Hounsell', H0=70., Om=0.3,
                 minRFphaseQual=-10, maxRFphaseQual=35):
        """
        class to estimate nsn + error

        Parameters
        ----------
        norm_factor : float, optional
            Simulation normalization factor. The default is 30.
        rate : str, optional
            SN rate production. The default is 'Hounsell'.
        H0 : float, optional
            H0 parameter value. The default is 70..
        Om : float, optional
            Om parameter value. The default is 0.3.
        minRFphaseQual : float, optional
            min Rest-Frame phase quality selection. The default is -10.
        maxRFphaseQual : float, optional
            max Rest-Frame phase quality selection. The default is 35.

        Returns
        -------
        None

        """

        from sn_tools.sn_rate import SN_Rate
        self.sn_rate = SN_Rate(rate=rate,
                               H0=H0,
                               Om0=Om,
                               min_rf_phase=minRFphaseQual,
                               max_rf_phase=maxRFphaseQual)
        self.norm_factor = norm_factor

    def __call__(self, data):
        """
        Method to estimate nsn and err_nsn (using multiproc)

        Parameters
        ----------
        data : pandas df
            Data to process.

        Returns
        -------
        res : pandas df
            output data.

        """

        hpixes = data['healpixID'].unique()

        params = {}
        params['data'] = data

        from sn_tools.sn_utils import multiproc

        res = multiproc(hpixes, params, self.nsn_multiproc, 8)

        return res

    def nsn_multiproc(self, toproc, params, j=0, output_q=None):
        """
        Method to estimate nsn using multiproc

        Parameters
        ----------
        toproc : list(int)
            list of healpixIDs to process.
        params : dict
            parameters.
        j : int, optional
            Internal tag for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            where to put the data. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        data = params['data']

        idx = data['healpixID'].isin(toproc)

        sel = data[idx]
        ccols = ['dbName', 'field', 'season', 'healpixID']
        res = sel.groupby(ccols).apply(
            lambda x: self.nsn_pixel(x), include_groups=False).reset_index()
        res['season'] = res['season'].astype(int)

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def nsn_pixel(self, grp):
        """
        Method to estimate nsn per pixel/season/field/dbName

        Parameters
        ----------
        grp : pandas df
            Data to process.

        Returns
        -------
        res : pandas df
            output data.

        """

        # grab season length and survey_area
        season_length = grp['season_length'].mean()
        survey_area = grp['survey_area'].mean()

        # observed number of SN
        nsn_obs = len(grp)

        # get expected number of SN from rate
        zmin = np.min(grp['z'])
        zmax = np.max(grp['z'])
        zz, rate, err_rate, nsn, err_nsn, age_univ = self.sn_rate(
            zmin=zmin, zmax=zmax,
            duration=season_length,
            survey_area=survey_area,
            account_for_edges=True, dz=0.001)

        if len(nsn) == 0:
            res = pd.DataFrame()
        else:
            nsn_exp = int(np.cumsum(nsn)[-1]*self.norm_factor)

            if nsn_exp < 1:
                nsn_exp = 1
            # get the variance (binomial)
            p = nsn_obs/nsn_exp
            if p > 1:
                # to account for statistical fluctuations in the production
                p = 1
            var_nsn = nsn_exp*p*(1-p)

            sigma_nsn = np.sqrt(var_nsn)

            nsn_obs = nsn_obs/self.norm_factor
            err_nsn_obs = sigma_nsn/self.norm_factor

            years = grp['year'].unique()
            r = []
            for year in years:
                idx = grp['year'] == year
                sel = grp[idx]
                frac_year = len(sel)/self.norm_factor/nsn_obs
                r.append((year, nsn_obs*frac_year, err_nsn_obs*frac_year))

            res = pd.DataFrame(r, columns=['year', 'nsn', 'err_nsn'])

        return res


def count_all(data, columns):
    """
    Function to estimate NSN and err_NSN from groupby (columns)

    Parameters
    ----------
    data : pandas df
        Data to process.
    columns : list(str)
        List of groupby columns.

    Returns
    -------
    tt : pandas df
        Result.

    """

    tt = data.groupby(columns).apply(lambda x: count(x)).reset_index()

    return tt


def count(grp):
    """
    Function to estimate nsn, err_nsn

    Parameters
    ----------
    grp : pandas df
        data to process.

    Returns
    -------
    res : pandas df
        Result.

    """

    dd = {}
    dd['nsn'] = [grp['nsn'].sum()]
    dd['err_nsn'] = [np.sqrt(grp['err_nsn']**2).sum()]

    res = pd.DataFrame.from_dict(dd)

    return res


def clean_level(tt):
    """
    Function to clean the level

    Parameters
    ----------
    tt : pandas df
        Data to process.

    Returns
    -------
    tt : pandas df
        cleaned df.

    """

    tt = tt[tt.columns.drop(list(tt.filter(regex='level')))]

    return tt


def get_nsn_new(data, norm_factor):
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
    idx = ddf['z'] >= 0.8
    sel = ddf[idx]
    resb = nsn(sel)
    resb = clean_level(resb)
    resb = resb.rename(columns={'nsn': 'nsn_z_08', 'err_nsn': 'err_nsn_z_08'})

    # get nsn - z >= 0.8 and sigmaC <= 0.04
    idx = ddf['z'] >= 0.8
    idx &= ddf['sigmaC'] <= 0.04
    sel = ddf[idx]
    resc = nsn(sel)
    resc = clean_level(resc)
    resc = resc.rename(
        columns={'nsn': 'nsn_z_08_sigmaC', 'err_nsn': 'err_nsn_z_08_sigmaC'})

    cols = ['dbName', 'field', 'healpixID', 'season', 'year']
    res_fi = resa.merge(resb, left_on=cols, right_on=cols, suffixes=['', ''])

    res_fi = res_fi.merge(resc, left_on=cols, right_on=cols, suffixes=['', ''])

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
parser.add_option('--ud_fields', type=str,
                  default='COSMOS,XMM-LSS',
                  help='UD fields to consider [%default]')
parser.add_option('--dd_fields', type=str,
                  default='CDFS,ELAISS1,EDFS_a,EDFS_b',
                  help='DD fields to consider [%default]')
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
ud_fields = opts.ud_fields.split(',')
dd_fields = opts.dd_fields.split(',')

# cumul = opts.cumul
# plot_moll = opts.plot_Mollweid

dataType = opts.dataType

# read config file
conf_df = pd.read_csv(config, comment='#')

# process data
ddf = process_DDF(conf_df, dataType, dbDir, runType,
                  timescale, timeslots, norm_factor)

print(ddf.columns)
df_nsn = get_nsn_new(ddf, norm_factor)

ccols = ['nsn_z_08', 'err_nsn_z_08']
print(df_nsn[ccols])
print(res)

idx = res['year'] <= 10

sel = res[idx]

stat = count_all(sel, ['dbName'])

print(stat)


# print(test)
# plot
# all fields
if 'nsn_all' in plots:
    fields = ud_fields+dd_fields
    plot_ddf_year(ddf, norm_factor, conf_df, nside=128,
                  cols=['year', 'dbName'],
                  fields=fields)
if 'nsn_ud' in plots:
    # UD only

    fields = ud_fields
    plot_ddf_year(ddf, norm_factor, conf_df, nside=128,
                  cols=['year', 'dbName'],
                  fields=fields)

plt.show()
