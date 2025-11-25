#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:15:04 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
from sn_telmodel.sn_atmosphere import Atmos_Transmission
import matplotlib.pyplot as plt
from optparse import OptionParser
from iminuit import Minuit


class Fit_Atmos:
    def __init__(self, airmass=1.0, beta=1.2,
                 fitparNames=['ozone', 'aerosol', 'pwv'],
                 fitparValues=[300., 0.05, 4.],
                 fitparLimits=[(100., 500.), (0., 1.), (0.5, 12.)]):
        """
        class to fit atmospheric parameters on a spectrum

        Parameters
        ----------
        airmass : float, optional
            airmass value. The default is 1.0.
        fitparNames : list(str), optional
            List of the names of the parameters to fit. 
            The default is ['ozone', 'aerosol', 'pwv'].
        fitparValues : list(float), optional
            list of initial values of parameters to fit. 
            The default is [300., 0.05, 4.].
        fitparLimits : list(pair(float)), optional
            parameter limits. The default is [(100.,500.),(0.,1.),(0.5,12.)].
        Returns
        -------
        None.

        """

        self.airmass = airmass
        self.beta = beta

        self.fitparNames = fitparNames
        self.fitparLimits = fitparLimits
        self.par_default = dict(zip(fitparNames, fitparValues))
        """
        dataValues = [data[key] for key in dataNames]
        for i, vals in enumerate(dataNames):
            exec('self.{} = dataValues[{}]'.format(vals, i))
        """
        # grab data from file
        self.atmos_trans_file = Atmos_Transmission(
            atmos_dir=atmosDir, atmos_type='from_file')
        self.atmos_trans_file.load_atmosphere(airmass=airmass,
                                              atmos_type='from_file')

        self.ndata = len(self.atmos_trans_file.atmosphere.wavelen)

    def __call__(self):
        """
        call function: where the fit is made

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        """

        params = []
        for pp in self.fitparNames:
            params.append(self.par_default[pp])

        res = self.minuit_fit(params)

        print(res)

        return res

    def minuit_fit(self, parameters):
        """
        fit using minuit

        Parameters
        ----------
        parameters : list(float)
            Initial parameter values.

        Returns
        -------
        dict_out : dict
            Fit results.

        """

        m = Minuit(self.xi_square, *parameters,
                   name=self.fitparNames)

        for i, vv in enumerate(self.fitparNames):
            limits = self.fitparLimits[i]
            m.limits[vv] = limits

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
        """
        the log likelihood to minimize

        Parameters
        ----------
        *parameters : list(float)
            parameter values.

        Returns
        -------
        Xmat : float
            the log likelihood value.

        """

        ozone = parameters[self.fitparNames.index('ozone')]
        pwv = parameters[self.fitparNames.index('pwv')]
        aerosol = parameters[self.fitparNames.index('aerosol')]

        if 'airmass' not in self.fitparNames:
            airmass = self.airmass
        else:
            airmass = parameters[self.fitparNames.index('airmass')]

        if 'beta' not in self.fitparNames:
            beta = self.beta
        else:
            beta = parameters[self.fitparNames.index('beta')]

        atmos_trans_obsatmo = Atmos_Transmission(atmos_type='obsatmo')
        atmos_trans_obsatmo.load_atmosphere(
            airmass=airmass, pwv=pwv, ozone=ozone, aerosol=aerosol, beta=beta)

        vva = self.atmos_trans_file.atmosphere.sb
        vvb = atmos_trans_obsatmo.atmosphere.sb

        sigma = 0.012
        Xmat = np.sum((vva-vvb)**2/sigma**2)

        return Xmat


parser = OptionParser(description='Script to plot a&tmos transmission')

parser.add_option('--atmosDir', type=str, default='atmos',
                  help='atmosphere location dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag version of the throughputs [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--aerosol', type=float, default=0.0,
                  help='aerosol value [%default]')
parser.add_option('--pwv', type=float, default=4.0,
                  help='precipitable water vapor value [%default]')
parser.add_option('--ozone', type=float, default=300.,
                  help='ozone value [%default]')
parser.add_option('--beta', type=float, default=1.4,
                  help='beta value [%default]')

opts, args = parser.parse_args()

atmosDir = 'throughputs_{}/{}'.format(opts.tag, opts.atmosDir)
airmass = opts.airmass
aerosol = opts.aerosol
pwv = opts.pwv
ozone = opts.ozone
beta = opts.beta

parNames = ['airmass', 'pwv', 'ozone', 'aerosol', 'beta']
parValues = [1.2, 4., 100, 0.1, 1.4]
parLimits = [(1., 3.), (0.5, 12.), (100., 500.), (0., 1.), (0., 5.)]

parNames = ['pwv', 'ozone', 'aerosol']
parValues = [4., 270, 0.1]
parLimits = [(0.5, 12.), (100., 500.), (0., 1.)]

myfit = Fit_Atmos(airmass, beta,
                  fitparNames=parNames,
                  fitparValues=parValues,
                  fitparLimits=parLimits)

resfit = myfit()


# from file
atmos_trans_file = Atmos_Transmission(
    atmos_dir=atmosDir, atmos_type='from_file')
atmos_trans_file.load_atmosphere(airmass=airmass, atmos_type='from_file')

# from getObsAtmo
pwv = resfit['pwv_fit']
ozone = resfit['ozone_fit']
aerosol = resfit['aerosol_fit']
atmos_trans_obsatmo = Atmos_Transmission(atmos_type='obsatmo')
atmos_trans_obsatmo.load_atmosphere(
    airmass=airmass, pwv=pwv, ozone=ozone, aerosol=aerosol)


params = {}
par_names = ['airmass', 'aerosol', 'pwv', 'ozone', 'beta', 'pressure']
par_plotnames = ['am', 'aer', 'pwv', 'ozone', 'beta', 'P']
pars = dict(zip(par_names, par_plotnames))
for key, vals in pars.items():
    sstr = 'params[\'{}\'] = atmos_trans_obsatmo.{}'.format(vals, key)
    exec(sstr)

ra = []
rb = []
for key, vals in params.items():
    ra.append(key)
    rb.append(vals)

ran = ','.join(ra)
rb = list(map(str, rb))
rbn = ','.join(rb)

# superimpose atmospheric transmission curves
labela = 'from file airmass({})+aerosol'.format(atmos_trans_file.airmass)
labelb = '({})=({})'.format(ran, rbn)

fig, ax = plt.subplots(figsize=(12, 8))
atmos_trans_file.plot_atmospheric_transmission(
    plt, fig=fig, ax=ax, label=labela)
atmos_trans_obsatmo.plot_atmospheric_transmission(
    plt, fig=fig, ax=ax, label=labelb, color='k', linestyle='dashed')

# residuals
vva = atmos_trans_file.atmosphere.sb
vvb = atmos_trans_obsatmo.atmosphere.sb

figb, axb = plt.subplots(figsize=(12, 8))
axb.plot(atmos_trans_file.atmosphere.wavelen, vva-vvb)

print(np.sum((vva-vvb)**2))

plt.show()
