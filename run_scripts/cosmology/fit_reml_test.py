#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:28:38 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from astropy.cosmology import w0waCDM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import gauss
from iminuit import Minuit


class REML_Fit:
    def __init__(self, data, dataNames,
                 fitparNames=['mubar', 'sigmaInt'],
                 fitparValues=[40, 0.12]):

        self.fitparNames = fitparNames
        dataValues = [data[key] for key in dataNames]
        for i, vals in enumerate(dataNames):
            exec('self.{} = dataValues[{}]'.format(vals, i))

        self.par_default = dict(zip(fitparNames, fitparValues))

        self.ndata = len(data)

    def __call__(self):

        params = []
        for pp in self.fitparNames:
            params.append(self.par_default[pp])

        res = self.minuit_fit(params)

        print(res)

    def minuit_fit(self, parameters):

        m = Minuit(self.xi_square, *parameters,
                   name=self.fitparNames)

        m.migrad()
        m.hesse()

        # grab the results: param values
        dict_out = {}
        res = m.values
        for name in self.fitparNames:
            dict_out['{}_fit'.format(name)] = res[name]
        fitpars = []
        for pp in self.fitparNames:
            fitpars.append(dict_out['{}_fit'.format(pp)])

        dict_out['Chi2_fit'] = myfit.xi_square(*fitpars)
        dict_out['NDoF'] = self.ndata-len(self.fitparNames)
        dict_out['Chi2_fit_red'] = dict_out['Chi2_fit']/dict_out['NDoF']
        # covariance matrix
        cov = m.covariance
        for i, vala in enumerate(self.fitparNames):
            for j, valb in enumerate(self.fitparNames):
                if j <= i:
                    dict_out['Cov_{}_{}_fit'.format(vala, valb)] = cov[i, j]

        return dict_out

    def xi_square(self, *parameters):

        mubar = parameters[self.fitparNames.index('mubar')]
        sigmaInt = parameters[self.fitparNames.index('sigmaInt')]

        cov_mu = self.sigma_mu**2+sigmaInt**2

        w = 1./cov_mu

        Xmat = np.sum(w*(self.mu_bin-mubar)**2)
        print('xx1', Xmat)
        Xmat -= np.sum(np.log(w))
        print('xx2', np.sum(np.log(w)))
        Xmat += np.log(np.sum(w))
        print('xx3', np.log(np.sum(w)))

        print('allo', mubar, sigmaInt, Xmat)
        return Xmat


H0 = 70.
Om0 = 0.3
w0 = -1.
wa = 0.
sigmaInt = 0.12

cosmology = w0waCDM(H0=H0, Om0=Om0, Ode0=1.-Om0, w0=w0, wa=wa)

z = np.arange(0.01, 1.1, 0.001)

df = pd.DataFrame(z, columns=['z'])
df['mu_th'] = cosmology.distmod(z).value
df['sigma_mu'] = np.random.normal(0., 0.00000005*df['mu_th'])
df['sigma_mu'] = np.abs(df['sigma_mu'])
# df['sigma_mu'] = 0.01
sigma_mu_int = np.sqrt(df['sigma_mu']**2+sigmaInt**2)

mu_shift = np.random.normal(0., sigma_mu_int)
df['mu'] = df['mu_th']+mu_shift

fig, ax = plt.subplots()

ax.errorbar(df['z'], df['mu'], yerr=df['sigma_mu'])

plt.show()

zfit = np.arange(0.2, 1.1, 0.05)

for i in range(len(zfit)-1):
    zmin = zfit[i]
    zmax = zfit[i+1]
    idx = df['z'] >= zmin
    idx &= df['z'] < zmax
    sel = pd.DataFrame(df[idx])
    print(zmin, zmax, len(sel))
    zzmin = sel['z'].min()
    zzmax = sel['z'].max()
    alpha = np.log(sel['z']/zzmin)/np.log(zzmax/zzmin)

    idx = np.abs(sel['z']-zzmin) < 1.e-5
    mu_b = sel[idx]['mu'].values[0]
    sig_b = sel[idx]['sigma_mu'].values[0]

    idx = np.abs(sel['z']-zzmax) < 1.e-5
    mu_b_plus = sel[idx]['mu'].values[0]
    sig_b_plus = sel[idx]['sigma_mu'].values[0]

    print(mu_b, mu_b_plus, alpha)
    sel['mu_bin'] = (1.-alpha)*mu_b+alpha*mu_b_plus
    # sel['sigma_mu'] = (1.-alpha)*sig_b+alpha*sig_b_plus

    print(sel[['mu', 'mu_bin']])

    # print(test)

    fitparValues = [sel['mu_bin'].mean(), sigmaInt]
    print('go', fitparValues, zmin, zmax, len(sel), sel['mu_bin'].mean())
    myfit = REML_Fit(sel, ['mu', 'mu_bin', 'sigma_mu'],
                     fitparValues=fitparValues)

    myfit()
    break
