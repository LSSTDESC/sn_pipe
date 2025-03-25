#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:48:17 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_tools.sn_utils import get_val
import pandas as pd
from sn_tools.sn_io import load_DataFrame
from sn_tools.sn_utils import n_z
from sn_analysis.sn_selection import selection_criteria
from sn_analysis.sn_calc_plot import select
from sn_plotter_analysis.sn_analyser_ddf import plot_versus
import numpy as np

from sn_tools.sn_rate import SN_Rate
from scipy.interpolate import interp1d
from sn_tools.sn_utils import multiproc


def effi(grp, sellist, bins=np.arange(0.0, 1.12, 0.02)):
    """
    Method to estimate efficiencies

    Parameters
    ----------
    grp : pandas df
        Data to process.
    sellist : list(sel)
        Selection list.
    bins : np array, optional
        z-bin range. The default is np.arange(0.0, 1.02, 0.02).

    Returns
    -------
    df_effi : pandas df
        efficiencies.

    """

    nsn_ref = n_z(grp, 'zmeas', bins=bins)

    # select data
    sel = select(grp, sellist)

    nsn_sel = n_z(sel, 'zmeas', bins=bins)

    df_t = nsn_ref.merge(nsn_sel, left_on=['zmeas'],
                         right_on=['zmeas'], suffixes=['_ref', '_sel'])

    effi = df_t['nsn_sel']/df_t['nsn_ref']
    err_effi = np.sqrt(df_t['nsn_sel']*(1.-effi))/df_t['nsn_ref']

    df_effi = pd.DataFrame(df_t['zmeas'].to_list(), columns=['zmeas'])
    df_effi['effi'] = effi
    df_effi['effi_err'] = err_effi

    df_effi = df_effi.fillna(0)

    del df_t
    del nsn_ref
    del nsn_sel

    return df_effi


def get_nsn(grp, sellist, zmin=0.01, zmax=1.2, dz=0.1):
    """
    Function to estimate the number of SNe Ia vs z
    from observing efficiency and SNe Ia rate explosion

    Parameters
    ----------
    grp : pandas df
        Data to process.
    zmin : float, optional
        min redshift. The default is 0.01.
    zmax : float, optional
        max redshift. The default is 1.1.
    dz : float, optional
        delta z. The default is 0.1.

    Returns
    -------
    None.

    """

    print('years', grp.groupby(['year']).size().reset_index(), len(grp))

    # get efficiencies
    effis = effi(grp, sellist)

    print(grp.name)

    # get snrates
    zplot = np.arange(zmin, zmax, dz)
    season_length = grp['season_length'].mean()
    survey_area = grp['survey_area'].mean()
    zz, rateInterp, rateInterp_err = getRates(zmin=zmin, zmax=zmax, dz=dz,
                                              survey_area=survey_area,
                                              season_length=season_length)
    # interpolate efficiency vs z
    effiInterp = interp1d(
        effis['zmeas'], effis['effi'], kind='linear',
        bounds_error=False, fill_value=0.)
    # interpolate variance efficiency vs z
    effiInterp_err = interp1d(
        effis['zmeas'], effis['effi_err'], kind='linear',
        bounds_error=False, fill_value=0.)

    nsn = effiInterp(zz)*rateInterp(zz)

    # get errors
    nsn_err = []
    for i in range(len(zz)):
        siga = effiInterp_err(zz[:i+1])*rateInterp(zz[:i+1])
        # sigb = effiInterp(zplot[:i+1])*rateInterp_err(zplot[:i+1])
        sigb = 0.
        nsn_err.append(np.sqrt(np.sum(siga**2 + sigb**2)))

    sn_df = pd.DataFrame(zz, columns=['zmeas'])
    sn_df['nsn'] = nsn
    sn_df['nsn_err'] = nsn_err

    nsn_bin = nsn_bin_err(sn_df)

    print(nsn_bin)
    # plot the results
    # plot_effi_nsn(zz, effiInterp, effiInterp_err, nsn, nsn_err)

    return nsn_bin


def nsn_bin_err(data, xvar='zmeas', yvar='nsn', yvar_err='nsn_err',
                bins=np.arange(0.0, 1.2, 0.1)):
    """
    Method to estimate nsn, err nsn per z-bin

    Parameters
    ----------
    data : pandas df
        Data to process.
    xvar : str, optional
        x-axis variable. The default is 'zmeas'.
    yvar : str, optional
        y-axis variable. The default is 'nsn'.
    yvar_err : str, optional
        y-axis variable error. The default is 'nsn_err'.
    bins : np array, optional
        z-bin values. The default is np.arange(0.0, 1.02, 0.2).

    Returns
    -------
    df : pandas df
        output data.

    """

    df = data.groupby(pd.cut(data[xvar], bins, right=False)).apply(
        lambda x: get_valpar(x))

    df = pd.DataFrame(df)

    print(df)
    _centers = (bins[:-1] + bins[1:])/2
    # df = df.drop(columns=[xvar])
    df[f'{xvar}_new'] = _centers

    return df


def get_valpar(grp, yvar='nsn', yvar_err='nsn_err'):
    """
    Method to grab sum and error NSN

    Parameters
    ----------
    grp : pandas df
        Data to process.
    yvar : str, optional
        var for the sum. The default is 'nsn'.
    yvar_err : str, optional
        var for the error. The default is 'nsn_err'.

    Returns
    -------
    pandas df
        output data.

    """

    dout = {}

    dout[yvar] = [grp[yvar].sum()]
    dout[yvar_err] = [np.sqrt((grp[yvar_err]**2).sum())]

    return pd.DataFrame.from_dict(dout)


def plot_effi_nsn(zz, effiInterp, effiInterp_err, nsn, nsn_err):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.errorbar(zz, effiInterp(zz), yerr=effiInterp_err(
        zz), linestyle='solid', marker='.', color='r')
    axb = ax.twinx()
    axb.errorbar(zz, nsn, yerr=nsn_err,
                 linestyle='None', marker='.', color='k')

    ax.grid(visible=True)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'observing efficiency')
    axb.set_ylabel(r'N$_{SN}$')
    ax.set_ylim([0, None])
    plt.show()


def getRates(rate='Hounsell', survey_area=9.6, season_length=180.,
             zmin=0.01, zmax=1.11, dz=0.01, H0=70., Om0=0.3):
    """
    Function to estimate SNe Ia rate explosion

    Parameters
    ----------
    rate : str, optional
        Rate to be used. The default is 'Hounsell'.
    survey_area : float, optional
        survey area. The default is 9.6.
    season_length : float, optional
        Season length. The default is 180..
    zmin : float, optional
        min redshift. The default is 0.01.
    zmax : float, optional
        max redshift. The default is 1.1.
    dz : float, optional
        delta redshift. The default is 0.01.
    H0 : float, optional
        Hubble constant. The default is 70..
    Om0 : float, optional
        Om0 cosmological parameter. The default is 0.3.

    Returns
    -------
    zz : list(float)
        redshift bins.
    rateInterp : list(float)
        Nsn per bin.
    rateInterp_err : list(float)
        Nsn error per bin.

    """

    rateSN = SN_Rate(rate=rate, H0=H0, Om0=Om0,
                     min_rf_phase=-15., max_rf_phase=30.)

    # estimate the rates and nsn vs z
    zz, rate, err_rate, nsn, err_nsn, age_universe = rateSN(zmin=zmin,
                                                            zmax=zmax,
                                                            dz=dz,
                                                            duration=season_length,
                                                            survey_area=survey_area,
                                                            account_for_edges=True)

    # rate interpolation
    rateInterp = interp1d(zz, nsn, kind='linear',
                          bounds_error=False, fill_value=0)
    rateInterp_err = interp1d(zz, err_nsn, kind='linear',
                              bounds_error=False, fill_value=0)

    return zz, rateInterp, rateInterp_err


parser = OptionParser(
    'Script to estimate the number of supernovae estimated fromm observing efficiency')

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell',
                  help="data dir[%default]")
parser.add_option("--zType", type=str,
                  default='spectroz', help="z type (spectroz/photz) [%default]")
parser.add_option("--listFields", type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                  help=" [%default]")
parser.add_option("--fieldType", type=str,
                  default='DDF',
                  help="field type [%default]")
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
parser.add_option('--dbList', type=str,
                  default='list_OS.csv',
                  help='list of OS to process [%default]')
parser.add_option('--selconfig', type=str,
                  default='G10_JLA',
                  help='selection [%default]')

opts, args = parser.parse_args()

dataDir = opts.dataDir
fieldType = opts.fieldType
runType = opts.runType
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
dataType = opts.dataType
dbList = opts.dbList
selconfig = opts.selconfig

# OS to process
dbNames = pd.read_csv(dbList, comment='#')

# load selection
sellist = selection_criteria()[selconfig]

# load data
for i, row in dbNames.iterrows():
    tt = 'load_{}(\'{}\',\'{}\',\'{}\',\'{}\',{},\'{}\')'.format(
        dataType, dataDir, row['dbName'], runType,
        timescale, timeslots, fieldType)
    print('rrr', tt)
    df = eval(tt)
    print(len(df))


idx = df['field'] == 'COSMOS'
idx &= df['healpixID'] == 108957
df = pd.DataFrame(df[idx])

# df.to_hdf('COSMOS.hdf5', key='sn')

nsn = df.groupby(['healpixID', 'season']).apply(
    lambda x: get_nsn(x, sellist)).reset_index()

nsn = nsn.drop(columns=['zmeas'])
nsn = nsn.rename(columns={'zmeas_new': 'zmeas'})
print('ezs', nsn)


"""
effis = df.groupby(['healpixID', 'year']).apply(
    lambda x: effi(x, sellist)).reset_index()

ccols = ['healpixID', 'survey_area', 'season_length']
effis = effis.merge(df[ccols], left_on=[
    'healpixID'], right_on=['healpixID'], suffixes=['', ''])

# plots
# effis.groupby(['healpixID']).apply(lambda x: plot_effis(x, timescale))

# get nsn

effis.groupby(['healpixID', timescale]).apply(lambda x: get_nsn(x))
"""
