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
                 fitparNames=['sigmaInt'],
                 fitparValues=[0.12]):

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

        dict_out['Chi2_fit'] = self.xi_square(*fitpars)
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

        # mubar = parameters[self.fitparNames.index('mubar')]
        sigmaInt = parameters[self.fitparNames.index('sigmaInt')]

        cov_mu = self.sigma_mu**2+sigmaInt
        # cov_mu = sigmaInt**2

        w = 1./cov_mu

        mubar = self.mu_SN.mean()
        vala = np.sum(w*(self.mu_SN-mubar)**2)
        valc = 2.*mubar*np.sum(w*(self.mu_SN-mubar))

        # vala /= cov_mu
        """
        valb = -0.5*np.sum(np.log(w))
        valc = 0.
        """
        valb = 0.5*(self.ndata-1)*np.log(np.sum(cov_mu))
        valb = np.sum(np.log(w))
        # valb = 0.5*(self.ndata-1)*np.log(cov_mu)
        """
        valc = np.log(np.sum(w))
        valc = 0.5*(self.ndata-1)*np.log(2*np.pi)
        valc = 0.5*np.sum(np.log(w))
        valc = 0.
        """
        Xmat = vala-valb+valc

        # print('allo', mubar, sigmaInt, Xmat, vala, valb, valc)
        # print(test)
        return Xmat


"""
H0 = 70.
Om0 = 0.3
w0 = -1.
wa = 0.
sigmaInt = 0.12

cosmology = w0waCDM(H0=H0, Om0=Om0, Ode0=1.-Om0, w0=w0, wa=wa)

z = np.arange(0.01, 1.1, 0.001)

df = pd.DataFrame(z, columns=['z'])
df['mu_th'] = cosmology.distmod(z).value
df['sigma_mu'] = np.random.normal(0., 0.005*df['mu_th'])
df['sigma_mu'] = np.abs(df['sigma_mu'])
# df['sigma_mu'] = 0.01
sigma_mu_int = np.sqrt(df['sigma_mu']**2+sigmaInt**2)

mu_shift = np.random.normal(0., sigma_mu_int)
df['mu'] = df['mu_th']+mu_shift

fig, ax = plt.subplots()

ax.errorbar(df['z'], df['mu'], yerr=df['sigma_mu'])

plt.show()
"""

# simple check on random gauss

mu_mean = 50.

sigma = 0.12

ndata = 100000

mu = np.random.normal(mu_mean, 0.50, ndata)
# mu = mu_mean
sigma_mu = 0.0
sigma_mu = np.random.normal(0., 0.50, ndata)
# sigma_mu = np.asarray([0.25]*ndata)
# sigma_mu = 0.02
sigmab = np.sqrt(sigma**2+sigma_mu**2)

print('sigmab)', sigmab)

rd = np.random.normal(mu, sigmab)


print('allo', rd)
rr = pd.DataFrame(rd, columns=['mu_SN'])
rr['sigma_mu'] = sigma_mu
fitparValues = [sigma**2]
myfit = REML_Fit(rr, ['mu_SN', 'sigma_mu'],
                 fitparValues=fitparValues)

myfit()

"""
mubar = rr['mu_SN'].mean()

sigma = np.sum((rr['mu_SN']-mubar)**2)
sigma /= len(rr)-1

print('hello', np.sqrt(sigma))
"""
print(test)


thefile = '../test_durvey/survey_sn_desc_ddf_gen_0.80_sn_v4.3.1_10yrs_desc_ddf_gen_0.80_sn_v4.3.1_10yrs_1_10_1_for_fit.hdf5'

sigmaInt = 0.12
df = pd.read_hdf(thefile)

zfit = np.arange(0.2, 1.1, 0.01)

for i in range(len(zfit)-1):
    zmin = zfit[i]
    zmax = zfit[i+1]
    idx = df['z_fit'] >= zmin
    idx &= df['z_fit'] < zmax
    sel = pd.DataFrame(df[idx])
    print(zmin, zmax, len(sel))
    zzmin = sel['z_fit'].min()
    zzmax = sel['z_fit'].max()
    alpha = np.log(sel['z_fit']/zzmin)/np.log(zzmax/zzmin)

    idx = np.abs(sel['z_fit']-zzmin) < 1.e-5
    mu_b = sel[idx]['mu_SN'].values[0]
    sig_b = sel[idx]['sigma_mu'].values[0]

    idx = np.abs(sel['z_fit']-zzmax) < 1.e-5
    mu_b_plus = sel[idx]['mu_SN'].values[0]
    sig_b_plus = sel[idx]['sigma_mu'].values[0]

    # print(mu_b, mu_b_plus, alpha)
    sel['mu_bin'] = (1.-alpha)*mu_b+alpha*mu_b_plus
    # sel['sigma_mu'] = (1.-alpha)*sig_b+alpha*sig_b_plus

    # print(sel[['mu', 'mu_bin']])

    # print(test)
    print('allo', (sel['sigma_mu']**2).sum())
    fitparValues = [sigmaInt**2]
    sel['sigma_mu'] = 0.
    print('go', fitparValues, zmin, zmax, len(sel), sel['mu_bin'].mean())
    myfit = REML_Fit(sel, ['mu_SN', 'mu_bin', 'sigma_mu'],
                     fitparValues=fitparValues)

    out_res = myfit()
    # break
