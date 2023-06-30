#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:59:56 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from sn_analysis.sn_selection import selection_criteria
from sn_analysis.sn_calc_plot import select, bin_it_mean
from sn_cosmology.cosmo_fit import CosmoFit, fom
import numpy as np
from astropy.cosmology import w0waCDM


class MyFit(CosmoFit):

    def fit_function(self,  *parameters):
        '''
        Calculates the function given at the end of the parameters.

        Parameters
        ----------
        *parameters : tuple of differents types of entries.
            It presents as : (a, b)
            a and b must be a numerical value.
        Returns
        -------
        f : array of numerical values
            Results of the array x in the funcion chosen in the parameters
            via the
            parameters given.
        '''

        # instance of the cosmology model
        to_import = 'from astropy.cosmology import {}'.format(self.cosmo_model)
        exec(to_import)

        import copy
        # set default parameters
        parDict = copy.deepcopy(self.cosmo_default)

        # change the parameters that have to be changes

        if len(parameters) > 0:
            # for i, val in enumerate(self.fitparNames):
            for vv in ['w0', 'wa', 'Om0']:
                try:
                    ind = self.fitparNames.index(vv)
                    parDict[vv] = parameters[ind]
                except Exception:
                    continue

        cosmo = eval(
            '{}(H0=70, Om0={}, Ode0=0.7,w0={}, wa={})'.format(self.cosmo_model,
                                                              parDict['Om0'],
                                                              parDict['w0'],
                                                              parDict['wa']))

        f = cosmo.distmod(self.z.to_list()).value
        self.h = np.max(f) * (10**-8)

        return f

    def xi_square(self, *parameters):
        '''
        Calculate Xi_square for a data set of value x and y and a function.
        Parameters
        ----------
        *parameters : tuple of differents types of entries.
            see the definition in function()
        Returns
        -------
        X : numerical value
            The Xi_square value.
        '''
        # Numerical calculations for each entry
        # X_mat = np.sum(((self.y-self.function(*parameters))**2)/self.sigma**2)
        # diag_sigma = np.linalg.inv(np.diag(self.sigma**2))
        if 'alpha' in self.fitparNames:
            alpha = parameters[self.fitparNames.index('alpha')]
            beta = parameters[self.fitparNames.index('beta')]
            Mb = parameters[self.fitparNames.index('Mb')]

            mu = self.mb+alpha*self.x1-beta*self.color-Mb
            var_mu = self.Cov_mbmb\
                + (alpha**2)*self.Cov_x1x1\
                + (beta**2)*self.Cov_colorcolor\
                + 2*alpha*self.Cov_x1mb\
                - 2*beta*self.Cov_colormb\
                - 2*alpha*beta*self.Cov_x1color

            sigma_mu = np.sqrt(var_mu)

            rind = []
            for vv in ['w0', 'wa', 'Om0']:
                try:
                    ind = self.fitparNames.index(vv)
                    rind.append(ind)
                except Exception:
                    continue
            idf = np.max(rind)+1
            fitparams = parameters[:idf]
            mu_th = self.fit_function(*fitparams)

        else:
            mu = self.mu
            mu_th = self.fit_function(*parameters)
            sigma_mu = self.sigma_mu
        denom = sigma_mu**2
        """
        if 'sigmaInt' in self.fitparNames:
            sigmaInt = parameters[self.fitparNames.index('sigmaInt')]
        """
        sigmaInt = 0.12
        denom += sigmaInt**2
        # print(var_mu)
        f = mu - mu_th
        # Matrix calculation of Xisquare
        # X_mat = np.matmul(f * f, sigma_mu**-2)
        X_mat = np.sum(f**2/denom)
        # prior to be set here

        if not self.prior.empty:
            idx = self.prior['varname'].isin(self.fitparNames)
            for io, row in self.prior[idx].iterrows():
                ref_val = row['refvalue']
                sigma_val = row['sigma']
                i = self.fitparNames.index(row['varname'])
                if len(parameters) > 0:
                    X_mat += ((parameters[i] - ref_val)**2)/sigma_val**2

        return X_mat


class Random_survey:
    def __init__(self, dataDir, dbName, statName, selconfig, seasons,
                 dict_sel,
                 survey=pd.DataFrame([('COSMOS', 1.1, 1.e8, 1, 10)],
                                     columns=['field', 'zmax', 'sigmaC',
                                              'season_min', 'season_max'])):
        """
        class to build a random survey

        Parameters
        ----------
        dataDir : str
            Location dir of the data.
        dbName : str
            OS name.
        statName : str
            DESCRIPTION.
        selconfig : str
            configuration selection.
        seasons : list(int)
            list of seasons to consider.
        dict_sel : str
            selection dict.
        survey : pandas df, optional
            List of fields to consider, zmax and sigmaC, season_min, season_max 
            The default is pd.DataFrame(['COSMOS', 1.1, 1.e8,1,10],
                                       columns=['field', 'zmax', 'sigmaC',
                                                'season_min','season_max'])

        Returns
        -------
        None

        """

        # grab data corresponding to dbName
        sndata = pd.read_hdf('{}/SN_{}.hdf5'.format(dataDir, dbName))
        # select seasons
        idx = sndata['season'].isin(seasons)
        sndata = sndata[idx]

        # select SN
        sndata = select(sndata, dict_sel)
        # print(len(sndata), np.unique(sndata['field']))

        # loading the number of expected SN/field/season

        nsn_field_season = pd.read_hdf('{}/{}'.format(dataDir, statName))
        nsn_field_season['NSN'] = nsn_field_season['NSN'].astype(int)
        idx = nsn_field_season['dbName'] == dbName
        idx &= nsn_field_season['selconfig'] == selconfig
        idx &= nsn_field_season['season'].isin(seasons)
        nsn_field_season = nsn_field_season[idx]

        print(nsn_field_season)

        # get the random survey
        self.data = self.random_sample(nsn_field_season, sndata, survey)

    def random_sample(self, nsn_field_season, sn_data, survey):
        """
        Function to extract a random sample of SN

        Parameters
        ----------
        nsn_field_season : pandas df
            reference data with eg NSN.
        sn_data : pandas df
            original df where data are extracted from.
        survey : pandas df
            list of field+zmac+sigmaC+season_min+season_max.

        Returns
        -------
        df_res : pandas df
            Resulting random sample.

        """

        df_res = pd.DataFrame()

        for i, row in nsn_field_season.iterrows():
            field = row['field']
            season = row['season']
            nsn = row['NSN']
            # get data
            idx = sn_data['field'] == field
            idx &= sn_data['season'] == season
            sel_sn = sn_data[idx]

            res = sel_sn.sample(n=nsn)

            # filter here
            idx = survey['field'] == field
            zmax = survey[idx]['zmax'].values[0]
            sigmaC = survey[idx]['sigmaC'].values[0]
            season_min = survey[idx]['season_min'].values[0]
            season_max = survey[idx]['season_max'].values[0]

            idb = res['z'] <= zmax
            idb &= res['sigmaC'] <= sigmaC
            idb &= res['season'] >= season_min
            idb &= res['season'] <= season_max

            df_res = pd.concat((df_res, res[idb]))

        return df_res


def analyze_data(data):
    """
    Function to analyze data

    Parameters
    ----------
    data : pandas df
        data to process.

    Returns
    -------
    None.

    """

    res = data.groupby(['field', 'season']).apply(
        lambda x: pd.DataFrame({'NSN': [len(x)]})).reset_index()

    resb = res.groupby(['field']).apply(
        lambda x: pd.DataFrame({'NSN': [x['NSN'].sum()]})).reset_index()

    outdict = {}
    nsn_tot = 0
    for i, row in resb.iterrows():
        field = row['field']
        nsn = row['NSN']
        outdict[field] = nsn
        nsn_tot += nsn

    outdict['all_Fields'] = nsn_tot
    return outdict


def plot_mu(data):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    df = bin_it_mean(data, xvar='z', yvar='mu')
    print(df)
    ax.errorbar(df['z'], df['mu'], yerr=df['mu_std'], marker='o',
                color='k', mfc='None', ms=5)
    cosmo = w0waCDM(H0=70,
                    Om0=0.3,
                    Ode0=0.7,
                    w0=-1., wa=0.0)

    bins = np.arange(df['z'].min(), 1.1, 0.02)
    f = cosmo.distmod(bins).value

    ax.plot(bins, f, 'bs', ms=5)

    df['diff'] = df['mu']-f
    figb, axb = plt.subplots()

    axb.plot(df['z'], df['diff'], 'ko')

    idx = df['z'] >= 0.5
    print('mean diff', np.mean(df[idx]['diff']))

    """
    idx = data['sigmaC'] <= 0.04
    seldata = data[idx]
    ax.plot(seldata['z'], seldata['mu'], 'r*', ms=5)

    figb, axb = plt.subplots()
    axb.plot(data['z'], data['sigma_mu'], 'ko', mfc='None', ms=5)
    """
    plt.show()


class HD_random:
    def __init__(self,
                 vardf=['z', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1',
                        'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
                        'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu'],
                 dataNames=['z', 'x1', 'color', 'mb', 'Cov_x1x1',
                            'Cov_x1color', 'Cov_colorcolor', 'Cov_mbmb',
                            'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu'],
                 fitconfig={}):

        self.vardf = vardf
        self.dataNames = dataNames
        self.fitconfig = fitconfig

    def __call__(self, data):

        dataValues = [data[key] for key in self.vardf]
        par_protect_fit = ['Om0']

        r = [('Om0', 0.3, 0.0073)]
        r.append(('sigmaInt', 0.12, 0.01))
        prior = pd.DataFrame(r, columns=['varname', 'refvalue', 'sigma'])

        dict_fits = {}
        for key, vals in self.fitconfig.items():

            fitparNames = list(vals.keys())
            fitparams = list(vals.values())
            print(key, fitparNames, fitparams)
            myfit = MyFit(dataValues, self.dataNames,
                          fitparNames=fitparNames, prior=prior,
                          par_protect_fit=par_protect_fit)

            dict_fit = myfit.minuit_fit(fitparams)
            fitpars = []
            for pp in fitparNames:
                fitpars.append(dict_fit['{}_fit'.format(pp)])
            dict_fit['Chi2_fit'] = myfit.xi_square(*fitpars)
            print(dict_fit)
            # fisher estimation
            # fisher_cov = myfit.covariance_fisher(fitparams)
            # print('Fisher', fisher_cov)
            print('')
            dict_fits[key] = dict_fit

        return dict_fits


def transform(dicta):

    dictb = {}

    for key, vals in dicta.items():
        dictb[key] = [vals]

    return dictb


# loading SN data
dataDir = 'SN_analysis_sigmaInt_0.12_Hounsell'
dbName = 'DDF_DESC_0.70_SN'
dbName = 'DDF_Univ_WZ'
statName = 'sn_field_season.hdf5'
selconfig = 'G10_JLA'
seasons = range(1, 6)
# seasons = range(2, 4)
dictsel = selection_criteria()[selconfig]
survey = pd.read_csv('input/DESC_cohesive_strategy/survey_scenario.csv')

"""
listDDF = 'COSMOS,XMM-LSS,ELAISS1,CDFS,EDFSa,EDFSb'
zmax = [1.1, 1.1, 0.6, 0.6, 0.6, 0.6]
sigmaC = [1.e8]*len(zmax)

ddf = pd.DataFrame(listDDF.split(','), columns=['field'])
ddf['zmax'] = zmax
ddf['sigmaC'] = sigmaC
"""


# plot_mu(data)

fitconfig = {}

fitconfig['fita'] = dict(zip(['w0', 'Om0', 'alpha', 'beta', 'Mb'],
                             [-1, 0.3, 0.13, 3.1, -19.08]))
fitconfig['fitb'] = dict(zip(['w0', 'wa', 'Om0', 'alpha', 'beta', 'Mb'],
                             [-1, 0.0, 0.3, 0.13, 3.1, -19.08]))
"""


fitconfig['fitc'] = dict(zip(['w0', 'Om0'],
                             [-1, 0.3]))
fitconfig['fitd'] = dict(zip(['w0', 'wa', 'Om0'],
                             [-1, 0.0, 0.3]))
"""
hd_fit = HD_random(fitconfig=fitconfig)

dict_res = {}
for i in range(10):
    data = Random_survey(dataDir, dbName, statName,
                         selconfig, seasons, dictsel,
                         survey=survey).data

    print('nsn', len(data))
    dict_ana = analyze_data(data)
    dict_ana['season'] = np.max(seasons)+1
    print(dict_ana)

    res = hd_fit(data)

    for key, vals in res.items():
        vals.update(dict_ana)
        res = pd.DataFrame.from_dict(transform(vals))
        if key not in dict_res.keys():
            dict_res[key] = pd.DataFrame()
        dict_res[key] = pd.concat((res, dict_res[key]))

print(dict_res)

"""
print(data.columns)
# cosmology

vardf = ['z', 'x1_fit', 'color_fit', 'mbfit', 'Cov_x1x1', 'Cov_x1color',
         'Cov_colorcolor',
         'Cov_mbmb', 'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu']
dataNames = ['z', 'x1', 'color', 'mb', 'Cov_x1x1', 'Cov_x1color', 'Cov_colorcolor',
             'Cov_mbmb', 'Cov_x1mb', 'Cov_colormb', 'mu', 'sigma_mu']

dataValues = [data[key] for key in vardf]

# fitparNames = ['w0', 'wa', 'Om0', 'alpha', 'beta', 'Mb']
fitparNames = ['w0', 'Om0', 'alpha', 'beta', 'Mb']
# params = [-1., 0., 0.3, 0.16, 3., -19.6]
params = [-1, 0.3, 0.13, 3.1, -19.08]


par_protect_fit = ['Om0']
# par_protect_fit = []
prior = pd.DataFrame([('Om0', 0.3, 0.0073)],
                     columns=['varname', 'refvalue', 'sigma'])
prior = pd.DataFrame()


myfit = MyFit(dataValues, dataNames,
              fitparNames=fitparNames, prior=prior,
              par_protect_fit=par_protect_fit)


dict_fit = myfit.minuit_fit(params)
fitpars = []
for pp in fitparNames:
    fitpars.append(dict_fit['{}_fit'.format(pp)])
dict_fit['Chi2_fit'] = myfit.xi_square(*fitpars)
# fitpars = []
# for pp in fitparNames:
#     fitpars.append(dict_fit['{}_fit'.format(pp)])
# dict_fit = {}
print(dict_fit)
fisher_cov = myfit.covariance_fisher(params)
dict_fit.update(fisher_cov)
# dict_fit['Chi2_fisher'] = myfit.xi_square(*[])
print(dict_fit)
"""
