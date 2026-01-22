#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 10:23:48 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_tools.sn_io import get_meta, Read_LightCurve
import numpy as np
import pandas as pd
import h5py
from astropy.table import Table, vstack
import matplotlib.pyplot as plt


def stat(grp):

    m5_mean = grp['fiveSigmaDepth'].mean()

    nvisits = len(grp)

    m5_coadd = 1.25*np.log10(np.sum(10**(0.8*grp['fiveSigmaDepth'])))

    m5_simp = m5_mean+1.25*np.log10(nvisits)

    res = {}

    res['fiveSigmaDepth'] = [m5_coadd]
    res['fiveSigmaDepth_nvisits'] = [m5_simp]

    return pd.DataFrame.from_dict(res)


def load_obs(ficha='obs_orig.npy', fichb='obs_coadd.npy'):

    obs_a = np.load(ficha, allow_pickle=True)

    obs_b = np.load(fichb, allow_pickle=True)

    dfa = pd.DataFrame.from_records(obs_a)

    dfb = pd.DataFrame.from_records(obs_b)

    return dfa, dfb


def load_params(paramFile):
    """
    Function to load simulation parameters

    Parameters
    ---------------
    paramFile: str
      name of the parameter file

    Returns
    -----------
    params: astropy table
    with simulation parameters

    """

    f = h5py.File(paramFile, 'r')
    print(f.keys(), len(f.keys()))
    params = Table()
    for i, key in enumerate(f.keys()):
        pars = Table.read(paramFile, path=key)
        params = vstack([params, pars])

    return params


def get_lc(lcpath, lcDir, lcName):

    lcs = Read_LightCurve(file_name=lcName, inputDir=lcDir)

    lc = lcs.get_table(lcpath)

    print(lcs)

    print('ooo', lcpath, np.unique(lc['filter']))
    # print(lc)

    return lc


def plot_lcs(dict_lcs, band='y'):

    fig, ax = plt.subplots(figsize=(12, 8))

    night = 542
    mmarks = dict(zip(range(4), ['o', 's', 'h', 'P']))
    i = -1
    for key, lc in dict_lcs.items():
        idx = lc['filter'] == band
        sel = lc[idx]
        i += 1
        ax.errorbar(sel['night'], sel['flux'],
                    yerr=sel['fluxerr'],
                    marker=mmarks[i], linestyle='None', mfc='None', label=key)
        print(sel[['filter', 'flux', 'fluxerr', 'snr', 'night']])
        idx &= lc['snr'] >= 1
        sela = lc[idx]
        """
        ax.errorbar(sela['night'], sela['flux'],
                    yerr=sela['fluxerr'], marker='*', linestyle='None')
        """
        idx = sel['night'] == night
        selb = sel[idx]
        print(selb[['filter', 'flux', 'fluxerr', 'snr']])
        ax.legend()
        ax.grid(visible=True)


def super_lcs_loop(metaTot, dir_dict):

    # get list of SNIDs

    list_SN = metaTot['SNID'].tolist()

    for lcpath in list_SN:

        super_lcs(metaTot, dir_dict, lcpath)


def super_lcs(metaTot, dir_dict, lcpath):

    # get list of SNIDs

    idx = metaTot['SNID'] == lcpath

    metadata = metaTot[idx]
    # print(metadata)
    # get lc
    lcDir = metadata['lc_dir'].value[0]
    lcName = metadata['lc_fileName'].value[0]
    print('allo', lcDir, lcName)

    dict_lcs = {}
    dict_lcs['obs_coadd'] = get_lc(lcpath, lcDir, lcName)
    for key, vals in dir_dict.items():
        dict_lcs[key] = get_lc(lcpath, vals, lcName)

    # print(lc.columns)
    # print(lcb[['band', 'band_cosmo', 'filter', 'night']])
    plot_lcs(dict_lcs)


def ana_lcs(metaTot, dir_dict, lcpath, band='y', night=440):

    # get list of SNIDs

    idx = metaTot['SNID'] == lcpath

    metadata = metaTot[idx]
    # print(metadata)
    # get lc
    lcDir = metadata['lc_dir'].value[0]
    lcName = metadata['lc_fileName'].value[0]

    dict_lcs = {}
    dict_lcs['obs_coadd'] = get_lc(lcpath, lcDir, lcName)
    for key, vals in dir_dict.items():
        dict_lcs[key] = get_lc(lcpath, vals, lcName)

    for key, lc in dict_lcs.items():
        print(key)
        idx = lc['night'] == night
        idx &= lc['filter'] == band
        sel = lc[idx]

        print(sel[['flux', 'fluxerr', 'time', 'snr']])


def get_pull_diff(dfb):

    df = pd.DataFrame(dfb)
    for vv in ['x1', 'color']:
        pullvar = 'pull_{}'.format(vv)
        fitvar = '{}_fit'.format(vv)
        sigvar = 'sigma_{}'.format(vv)
        df[pullvar] = (df[fitvar]-df[vv])/df[sigvar]

    return df


def select_df(df):

    df['sigma_x1_new'] = np.sqrt(df['Cov_x1x1'])
    df['sigma_color'] = np.sqrt(df['Cov_colorcolor'])

    idx = df['fitstatus'] == 'fitok'

    return pd.DataFrame(df[idx])


def compare_sn(dict_sn):

    dict_sn_sel = {}
    for key, vals in dict_sn.items():
        dfa = select_df(vals)
        dict_sn_sel[key] = get_pull_diff(dfa)

    """
    sn_obs_coadd = select_df(sn_obs_coadd)
    sn_lc_coadd = select_df(sn_lc_coadd)
    """
    """
    tt = sn_obs_coadd.merge(sn_lc_coadd, left_on=['SNID'], right_on=[
                            'SNID'], suffixes=['_obs_coadd', '_lc_coadd'])

    """
    """
    sn_obs_coadd = get_pull_diff(sn_obs_coadd)
    sn_lc_coadd = get_pull_diff(sn_lc_coadd)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    xmin = -5
    xmax = 5
    bins = np.arange(xmin, xmax, 0.2)
    pullvar = 'pull_color'

    for key, vals in dict_sn_sel.items():
        ax.hist(vals[pullvar], bins=bins,
                histtype='step', label=key)
        idx = vals[pullvar] >= xmin
        idx &= vals[pullvar] <= xmax
        # idx &= vals['z'] < 0.5
        sel = vals[idx]
        print(key, sel[pullvar].median(), sel[pullvar].std(), len(sel))

    """
    ax.hist(sn_lc_coadd[pullvar], bins=bins, histtype='step', label='lc coadd')

    idx = sn_lc_coadd['pull_color'] >= 5.
    idx &= sn_lc_coadd['pull_color'] <= 10

    print(sn_lc_coadd[idx]['SNID'])
    """

    ax.legend()
    plt.show()


dfa, dfb = load_obs()

"""
dfc = dfa.groupby(['healpixID', 'filter', 'night']).apply(
    lambda x: stat(x), include_groups=True).reset_index()

print(len(dfc), len(dfb))

print(dfc)

print(dfb)
"""
dir_sn_a = '../test_simu_obscoadd_1_smearf_1'
dir_sn_a = '../test_simu_obscoadd_1_smearf_0_lccoadd_0'
dir_sn_b = '../test_simu_obscoadd_0_smearf_0_lccoadd_1'
dir_sn_a = '../test_simu_obscoadd_1_smearf_1_lccoadd_0'
dir_sn_b = '../test_simu_obscoadd_0_smearf_1_lccoadd_1_new'
dir_sn_c = '../test_simu_obscoadd_0_smearf_0_lccoadd_0'
fich_simu_a = 'Simu_prod_COSMOS_2_7.hdf5'
fich_sn = 'SN_prod_COSMOS_2.hdf5'

# load SN metadata
metaTot = get_meta('prod_COSMOS_2_7', '', dir_sn_a)


# load SN data
dict_sn = {}
dict_sn['obs_coadd'] = pd.read_hdf('{}/{}'.format(dir_sn_a, fich_sn))
dict_sn['lc_coadd'] = pd.read_hdf('{}/{}'.format(dir_sn_b, fich_sn))
dict_sn['no_coadd'] = pd.read_hdf('{}/{}'.format(dir_sn_c, fich_sn))
# compare SN values
compare_sn(dict_sn)
"""
# plot LCs (superimposed)
dir_dict = {}
dir_dict['lc_coadd'] = dir_sn_b
dir_dict['no_coadd'] = dir_sn_c
super_lcs(metaTot, dir_dict, 'SN_0108958_02_00010_4')
ana_lcs(metaTot, dir_dict, 'SN_0108958_02_00010_4')
"""
plt.show()
