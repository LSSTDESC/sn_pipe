#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:20:30 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from sn_analysis.sn_calc_plot import bin_it, bin_it_mean
from optparse import OptionParser
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sn_analysis.sn_tools import complete_df
from scipy.interpolate import make_interp_spline
from sn_analysis.sn_nsn_effi import getRates
from scipy.interpolate import interp1d


def load_data(dbDir, dbName, runType, timescale, seasons, alpha=0.13, beta=3.1):
    """
    Function to load data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        dbname.
    runType : str
        run type.
    timescale : str
        time scale.
    seasons : list(int)
        List of seasons.

    Returns
    -------
    df : pandas df
        Data for plot.

    """

    mainDir = '{}/{}/{}'.format(dbDir, dbName, runType)

    df = pd.DataFrame()
    for seas in seasons:
        path = '{}/*_{}_{}.hdf5'.format(mainDir, timescale, seas)
        print('path', path)
        fis = glob.glob(path)
        for fi in fis:
            print('loading', fi)
            tt = pd.read_hdf(fi)
            df = pd.concat((df, tt))

        # pull estimation
        df['pull_x1'] = (df['x1']-df['x1_fit'])/df['sigmax1']
        df['pull_c'] = (df['color']-df['color_fit'])/df['sigma_c']
        df['pull_daymax'] = (df['daymax']-df['t0_fit'])/df['sigma_t0']
        df['diff_x1'] = (df['x1']-df['x1_fit'])
        df['diff_c'] = (df['color']-df['color_fit'])
        df['chisq_ndof'] = df['chisq']/df['ndof']
        df['diff_x1_c'] = alpha*df['diff_x1']-beta*df['diff_c']

    return df


def gauss(x, *p):
    """
    gaussian function

    Parameters
    ----------
    x : float
        x values.
    *p : list(float)
        gaussian parameters.

    Returns
    -------
    list(float)
        function values.

    """
    A, mu, sigma = p
    return A/np.sqrt(sigma)*np.exp(-(x-mu)**2/(2.*sigma**2))


def plot_pull(dfa, pullvar, figtitle='', fitgauss=True):
    """
    Function to plot and fit of a pull distribution

    Parameters
    ----------
    dfa : pandas df
        Data container.
    pullvar : str
        pull variable.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots()
    fig.suptitle(figtitle)
    # ax.hist(dfa['pull_x1'], histtype='step')
    idx = np.abs(dfa[pullvar]) <= 5
    sel = dfa[idx]
    print(len(sel)/len(dfa))
    ax.hist(sel[pullvar], histtype='step', bins=80)

    # Get the fitted curve
    if fitgauss:
        coeff = fit_pull(sel, pullvar)
        xmin = sel[pullvar].min()
        xmax = sel[pullvar].max()
        newbins = np.arange(xmin, xmax, 0.01)
        hist_fit = gauss(newbins, *coeff)
        mean = np.round(coeff[1], 2)
        sigma = np.round(coeff[2], 2)
        leg = 'pull= {} +- {}'.format(mean, sigma)
        ax.plot(newbins, hist_fit, label=leg)
        print('bbb', coeff[0], coeff[1], coeff[2])
    print(figtitle, np.mean(sel[pullvar]))
    """
    ttb = gauss(bin_centres, *coeff)
    print('chi2', np.sum(np.power(hist-ttb, 2))/(len(bin_centres)-3))
    print(hist)
    print(ttb)
    print(hist-ttb)
    """
    ax.grid(visible=True)
    ax.legend()


def fit_pull(sel, pullvar):

    hist, bins = np.histogram(sel[pullvar], bins=80)
    bin_centres = (bins[:-1] + bins[1:])/2
    p0 = [1., 0., 1.]
    try:
        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
    except Exception:
        coeff = [-1, -1, -1]

    return coeff


def plot_hist(dfa, var, figtit='', fig=None, ax=None, label='', bins=10):
    """
    Function to plot hist

    Parameters
    ----------
    dfa : pandas df
        Data container.
    var : str
        Variable to plot.

    Returns
    -------
    None.

    """

    if fig is None:
        fig, ax = plt.subplots()
    if figtit != '':
        fig.suptitle(figtit)
    ax.hist(dfa[var], histtype='step', bins=bins, label=label)


def plot_nsn_hist(dfa):
    ccols = ['healpixID', 'RA', 'Dec']
    print(np.unique(dfa['healpixID']))
    dfb = dfa.groupby(['healpixID']).size().reset_index(name='nsn')
    print(dfb)
    plt.hist(dfb['nsn'], histtype='step')


def plot_all_pull(df, seasons):

    zmin = 0.5
    zmax = 1.1
    dz = 0.01
    zrange = np.arange(zmin, zmax+dz, dz)

    print('seasons', seasons)
    for seas in seasons:
        idx = df['season'] == seas
        sela = df[idx]
        for z in zrange:
            idxa = sela['z'] >= z
            idxa &= sela['z'] < z+dz
            sel = sela[idxa]
            print('pull man', len(sel))
            if len(sel) < 1:
                continue
            plot_pull(sel, 'pull_x1',
                      figtitle='pull x1 - season {}'.format(seas))
            plot_pull(sel, 'pull_c',
                      figtitle='pull color - season {}'.format(seas))
            plot_pull(sel, 'pull_daymax',
                      figtitle='pull daymax - season {}'.format(seas))
            plt.show()


def fit_all_pulls_allz(df):

    zmin = 0.0
    zmax = 1.1
    dz = 0.06
    zrange = np.arange(zmin, zmax+dz, dz)
    pullvars = ['x1', 'c']

    pullvars_str = list(map(lambda x: 'pull_'+x, pullvars))
    rt = []
    for z in zrange:
        idxa = df['z'] >= z
        idxa &= df['z'] < z+dz
        sel = df[idxa]
        print('pull man', len(sel), z, z+dz)
        if len(sel) < 2:
            continue
        r = [z, z+dz]
        for pullvar in pullvars_str:
            idx = np.abs(sel[pullvar]) <= 5
            selb = sel[idx]
            coeff = fit_pull(selb, pullvar)
            mean = np.round(coeff[1], 3)
            sigma = np.round(coeff[2], 3)
            r += [mean, sigma]

        rt.append(r)

    columns = ['zmin', 'zmax']

    for pullvar in pullvars:
        columns += ['mean_{}'.format(pullvar), 'sigma_{}'.format(pullvar)]

    res = pd.DataFrame(rt, columns=columns)

    return res


def KS_proba(df):

    zmin = 0.25
    zmax = 1.1
    dz = 0.02
    zrange = np.arange(zmin, zmax+dz, dz)
    ks_vars = ['x1_fit', 'color_fit']
    from scipy import stats
    distrib_ref = {}
    r = []

    for vv in ks_vars:
        df = df.round({'{}'.format(vv): 2})
    for z in zrange:
        idxa = df['z'] >= z
        idxa &= df['z'] < z+dz
        idxa &= df['x1'] >= -2
        idxa &= np.abs(df['color']) <= 0.2

        sel = df[idxa]
        if len(sel) < 5:
            continue
        print('there man', z, z+dz, len(sel))
        # grab (x1,c) distributions
        ro = [z, z+dz]
        distrib_current = {}
        for vv in ks_vars:
            # plt.hist(sel[vv], histtype='step')
            # plt.show()
            distrib_current[vv] = sel[vv]

        if distrib_ref:
            for vv in ks_vars:
                res = stats.ks_2samp(
                    distrib_ref[vv], distrib_current[vv], keepdims=True)
                print(vv, res.pvalue)
                """
                fig, ax = plt.subplots()
                ax.hist(distrib_ref[vv], bins=20, histtype='step')
                ax.hist(distrib_current[vv], bins=20, histtype='step')
                plt.show()
                """
                ro.append(res.pvalue)
        r.append(ro)
        if np.abs(z-zmin) < 1.e-3:
            distrib_ref = distrib_current

    columns = ['zmin', 'zmax']

    for vv in ks_vars:
        columns += ['pvalue_{}'.format(vv)]

    res = pd.DataFrame(r, columns=columns)

    return res


def plot_vs(data, varx='chisq_ndof', vary='diff_x1'):

    fig, ax = plt.subplots()

    ax.plot(data[varx], data[vary], 'ko')


def get_zlim(grp, sigmaC_ref=0.04, plotIt=False):

    dz = 0.05
    bins = np.arange(0.01, 1.1+dz, dz)
    df = bin_it_mean(grp, xvar='zmeas', yvar='sigmaC', bins=bins)

    df['sigmaC_plus'] = df['sigmaC']+df['sigmaC_std']
    df['sigmaC_minus'] = df['sigmaC']-df['sigmaC_std']

    zlim = interp1d(df['sigmaC'], df['zmeas'],
                    bounds_error=False, fill_value=0.)
    zlim_plus = interp1d(df['sigmaC_minus'], df['zmeas'],
                         bounds_error=False, fill_value=0.)
    zlim_minus = interp1d(df['sigmaC_plus'], df['zmeas'],
                          bounds_error=False, fill_value=0.)

    zlim = zlim(sigmaC_ref)
    zlim_p = zlim_plus(sigmaC_ref)
    zlim_m = zlim_minus(sigmaC_ref)

    res = pd.DataFrame([zlim], columns=['zlim'])
    res['zlim_p'] = zlim_p
    res['zlim_m'] = zlim_m

    return res

    if plotIt:
        fig, ax = plt.subplots()

        # ax.errorbar(df['zmeas'], df['sigmaC'], yerr=df['sigmaC_std'])
        ax.fill_between(df['zmeas'], df['sigmaC_plus'],
                        df['sigmaC_minus'], color='yellow')
        ax.grid(visible=True)

        fig, ax = plt.subplots()
        ax.plot(df['sigmaC'], df['zmeas'])
        ax.plot(df['sigmaC_plus'], df['zmeas'])

        ax.plot(df['sigmaC_minus'], df['zmeas'])

        ax.grid(visible=True)
        plt.show()


def get_nsn(grp, norm_factor=200, zmin=0.1, zmax=1.1, dz=0.01):

    norm_factor_bin = norm_factor*dz/0.01
    bins = np.arange(zmin, zmax+dz, dz)
    print('aoo', grp)
    effis = bin_it(grp, xvar='zmeas', norm_factor=norm_factor_bin,
                   bins=bins, outvar='effi')

    print(effis)
    hpix = int(grp['healpixID'].unique()[0])
    fig, ax = plt.subplots()
    fig.suptitle(hpix)
    ax.errorbar(effis['zmeas'], effis['effi'], yerr=effis['effi_err'])

    # plt.show()

    zmin = np.min(bins)
    zmax = np.max(bins)

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

    df_nsn = pd.DataFrame(zz, columns=['zmeas'])
    df_nsn['nsn'] = rateInterp(zz)
    df_nsn['nsn_effi'] = nsn
    df_nsn['nsn_effi_err'] = nsn_err

    fig, ax = plt.subplots()

    tp = np.cumsum(df_nsn['nsn_effi'].to_list())
    print(tp, type(tp))
    nsn = tp[-1]

    tpb = np.cumsum(df_nsn['nsn'].to_list())
    print(tp, type(tp))

    nsnb = tpb[-1]

    ax.plot(df_nsn['zmeas'], tpb/nsn-tp/nsnb)

    # ax.plot(df_nsn['zmeas'], tpb/nsnb)
    """

    ax.plot(df_nsn['zmeas'], (tp/nsnb)/(tpb/nsnb))
    """
    ax.grid()
    plt.show()


def plot_mu_z(tt):

    print(tt.columns)
    dz = 0.1
    bins = np.arange(0.01, 1.1+dz, dz)
    hpixes = tt['healpixID'].unique()
    xvar = 'zmeas'
    yvar = 'diff_x1_c'
    for hpix in hpixes:
        fig, ax = plt.subplots()
        ijk = tt['healpixID'] == hpix
        sel = tt[ijk]
        # sel = clean_bins(sel, xvar=xvar, yvar=yvar, bins=bins)
        # ax.plot(sel['zmin'], sel['pvalue_color_fit'])
        bb = bin_it_mean(sel, xvar=xvar, yvar=yvar, bins=bins)
        ax.errorbar(bb[xvar], bb[yvar], yerr=bb['{}_std'.format(yvar)])

        """
        xnew = np.linspace(np.min(sel['z']), np.max(sel['z']), 100)
        spl = make_interp_spline(sel['z'], sel['NSN'], k=5)  # type: BSpline
        spl_smooth = spl(xnew)
        ax.plot(xnew, spl_smooth)
        """
        ax.grid()
        plt.show()


def clean_bins(grp, xvar='zmeas', yvar='diff_mu', bins=np.arange(0.01, 1.11, 0.01), nsigma=5.):

    bb = bins.tolist()

    res = pd.DataFrame()
    for i in range(len(bb)-1):
        bxa = bb[i]
        bxb = bb[i+1]
        idx = grp[xvar] >= bxa
        idx &= grp[xvar] < bxb

        sel_bin = grp[idx]

        mean = sel_bin[yvar].mean()
        std = sel_bin[yvar].std()

        idxb = sel_bin[yvar] >= mean-nsigma*std
        idxb &= sel_bin[yvar] <= mean+nsigma*std

        res = pd.concat((res, sel_bin[idxb]))

    return res


parser = OptionParser(description='Script to analyze SN prod')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_WFD_sigmaInt_0.0_Hounsell_z_smflux_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v3.4_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='WFD_spectroz_nosat',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale [%default]')
parser.add_option('--seasons', type=str,
                  default='1',
                  help='seasons/years to process [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS',
                  help='fields to process [%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName.split(',')
runType = opts.runType
timescale = opts.timescale
seasons = opts.seasons.split(',')
seasons = list(map(int, seasons))
fields = opts.fields.split(',')

dfa = pd.DataFrame()
print('kkkkkk', dbName)
for dbNam in dbName:
    dff = load_data(dbDir, dbNam, runType, timescale, seasons)
    dff['dbName'] = dbNam
    dfa = pd.concat((dfa, dff))

dfa = complete_df(dfa)

ninit = len(dfa)

idx = dfa['field'].isin(fields)
nsn = len(dfa[idx])
"""


idx = dfa['sigma_c'] <= 0.04
# idx &= dfa['Nfilt_10'] > 2
"""
# idx &= dfa['sigma_c'] <= 0.04
"""
idx &= dfa['n_epochs_m10_p5'] >= 5
idx &= dfa['n_epochs_phase_minus_10'] >= 2
idx &= dfa['n_epochs_bef'] >= 5
idx &= dfa['n_epochs_aft'] >= 10
"""
# idx &= dfa['n_epochs_phase_minus_10'] >= 3
# idx &= dfa['sigmax1'] <= 0.10
# idx &= dfa['sigma_mu'] <= 0.2
# idx &= dfa['n_epochs_bef'] >= 5
# idx &= dfa['n_epochs_aft'] >= 10
# idx &= dfa['n_epochs_phase_plus_20'] > 3
# idx &= dfa['n_epochs_phase_minus_10'] >= 3
# idx &= (dfa['Nfilt_10'] >= 2) | (dfa['Nfilt_20'] >= 1)
# idx &= (dfa['Nfilt_2'] >= 3)
# idx &= (dfa['Nfilt_10'] >= 2)

dfa = dfa[idx]

"""
tt = dfa.groupby(['field', 'healpixID', 'season']).apply(
    lambda x: KS_proba(x)).reset_index()


tt = dfa.groupby(['field', 'healpixID', 'season']).apply(
    lambda x: fit_all_pulls_allz(x)).reset_index()
"""

print(dfa.columns.to_list())

"""
print('again', dfa['diff_mu'])
plot_mu_z(dfa)
print(test)
"""

tt = dfa.groupby(['field', 'healpixID', 'season']).apply(
    lambda x: get_zlim(x)).reset_index()

tt.to_hdf('res_new.hdf5', key='zlim')
print(tt)

print(test)
hpixes = tt['healpixID'].unique()

for hpix in hpixes:
    fig, ax = plt.subplots()
    ijk = tt['healpixID'] == hpix
    sel = tt[ijk]
    # ax.plot(sel['zmin'], sel['pvalue_color_fit'])
    ax.plot(sel['z'], sel['NSN'])
    xnew = np.linspace(np.min(sel['z']), np.max(sel['z']), 100)
    spl = make_interp_spline(sel['z'], sel['NSN'], k=5)  # type: BSpline
    spl_smooth = spl(xnew)

    ax.plot(xnew, spl_smooth)

    ax.grid()
    plt.show()

ax.grid()
# plt.show()

"""
plot_hist(dfa, 'diff_mu', bins=100)

nsn_filt = len(dfa)

print('filt', nsn_filt/nsn)
"""

print(dfa['diff_mu'].mean(), dfa['diff_mu'].std())
plt.show()


plot_all_pull(dfa, seasons)

"""
print(dfa.columns)
varx = 'n_epochs_phase_plus_20'
plot_vs(dfa, varx='SNR')
plot_vs(dfa, varx='Nfilt_5', vary='diff_c')
"""
plt.show()

print(dfa.columns, len(dfa)/30.)


plot_pull(dfa, 'pull_x1', figtitle='pull x1')
plot_pull(dfa, 'pull_c', figtitle='pull color')
"""
plot_pull(dfa, 'diff_x1', figtitle='diff x1', fitgauss=False)
plot_pull(dfa, 'diff_c', figtitle='diff color', fitgauss=False)
"""
# plot_pull(dfa, 'pull_daymax')
"""
ccols = ['n_epochs_bef', 'n_epochs_aft', 'Nfilt_10', 'Nfilt_15', 'Nfilt_20',
         'n_epochs_phase_minus_10',
         'n_epochs_phase_plus_20', 'n_epochs_m10_p35', 'n_epochs_m10_p5',
         'n_epochs_p5_p20', 'n_bands_m8_p10']
# ccols = ['sigmax1', 'sigmaC', 'sigmat0']
for vv in ccols:
    fig, ax = plt.subplots()
    for dbNam in dbName:
        idx = dfa['dbName'] == dbNam
        sel = dfa[idx]
        plot_hist(sel, vv, figtit=vv, fig=fig, ax=ax, label=dbNam, bins=15)
    ax.legend()
"""
plt.show()
