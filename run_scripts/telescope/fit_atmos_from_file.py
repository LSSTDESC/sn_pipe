#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:15:04 2024

@author: philippe.gris@clermont.in2p3.fr
"""


from sn_telmodel.sn_atmosphere import Atmos_Transmission
from sn_tools.sn_io import checkDir

import matplotlib.pyplot as plt
from optparse import OptionParser
from iminuit import Minuit
import pandas as pd
import numpy as np


class Fit_Atmos:
    def __init__(self, params,
                 fitparNames=['ozone', 'aerosol', 'pwv'],
                 fitparValues=[300., 0.05, 4.],
                 fitparLimits=[(100., 500.), (0., 1.), (0.5, 12.)]):
        """
        class to fit atmospheric parameters on a spectrum

        Parameters
        ----------
        params : dict.
            parameter values.
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

        self.params = params

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

        ppfit = {}
        # get fit parameters values
        for key in self.fitparNames:
            ppfit[key] = parameters[self.fitparNames.index(key)]

        # complete with fixed parameters
        lla = self.params.keys()
        bb = list(set(self.params.keys()) - set(self.fitparNames))
        ppfit_b = {k: self.params[k] for k in bb}

        # merge dicts
        ppfit = ppfit | ppfit_b

        atmos_trans_obsatmo = Atmos_Transmission(atmos_type='obsatmo')
        atmos_trans_obsatmo.load_atmosphere(
            airmass=ppfit['airmass'], pwv=ppfit['pwv'],
            ozone=ppfit['ozone'], aerosol=ppfit['aerosol'], beta=ppfit['beta'])

        vva = self.atmos_trans_file.atmosphere.sb
        vvb = atmos_trans_obsatmo.atmosphere.sb

        sigma = 0.012
        Xmat = np.sum((vva-vvb)**2/sigma**2)

        return Xmat


def plot_atmos_trans(atmosDir, resfit, params, fitparNames):
    """
    Function to plot atmos transmission

    Parameters
    ----------
    atmosDir : str
        atmos dir.
    resfit : dict
        atmos params for getObsAtmo.

    Returns
    -------
    None.

    """
    listpars = params.keys()

    for ll in listpars:
        llfit = '{}_fit'.format(ll)
        if llfit in resfit.keys():
            params[ll] = resfit[llfit]

    # from file
    atmos_trans_file = Atmos_Transmission(
        atmos_dir=atmosDir, atmos_type='from_file')
    atmos_trans_file.load_atmosphere(
        airmass=params['airmass'], atmos_type='from_file')

    # from getObsAtmo
    atmos_trans_obsatmo = Atmos_Transmission(atmos_type='obsatmo')
    atmos_trans_obsatmo.load_atmosphere(
        airmass=params['airmass'],
        pwv=params['pwv'],
        ozone=params['ozone'],
        aerosol=params['aerosol'],
        beta=params['beta'])

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
        rb.append(np.round(vals, 3))

    ran = ','.join(ra)
    rb = list(map(str, rb))
    rbn = ','.join(rb)

    # superimpose atmospheric transmission curves
    labela = 'from file airmass({})+aerosol'.format(atmos_trans_file.airmass)
    labelb = '({})=({})'.format(ran, rbn)

    fig, ax = plt.subplots(figsize=(15, 8))
    atmos_trans_file.plot_atmospheric_transmission(
        plt, fig=fig, ax=ax, label=labela)
    atmos_trans_obsatmo.plot_atmospheric_transmission(
        plt, fig=fig, ax=ax, label=labelb, color='k', linestyle='dashed')

    # residuals
    vva = atmos_trans_file.atmosphere.sb
    vvb = atmos_trans_obsatmo.atmosphere.sb

    figb, axb = plt.subplots(figsize=(12, 8))
    axb.plot(atmos_trans_file.atmosphere.wavelen, (vva-vvb)/vva)
    axb.grid(visible=True)

    plt.show()


parser = OptionParser(description='Script to plot a&tmos transmission')

parser.add_option('--atmosDir', type=str, default='atmos',
                  help='atmosphere location dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag version of the throughputs [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--aerosol', type=float, default=0.0,
                  help='aerosol value [%default]')
parser.add_option('--pwv', type=float, default=5.0,
                  help='precipitable water vapor value [%default]')
parser.add_option('--ozone', type=float, default=300.,
                  help='ozone value [%default]')
parser.add_option('--beta', type=float, default=1.4,
                  help='beta value [%default]')
parser.add_option('--plot_results', type=int, default=0,
                  help='to plot fit results [%default]')
parser.add_option('--outName', type=str, default='None',
                  help='output file name [%default]')
parser.add_option('--outDir', type=str, default='../fit_lsst_atmos',
                  help='output dir [%default]')

opts, args = parser.parse_args()

atmosDir = 'throughputs_{}/{}'.format(opts.tag, opts.atmosDir)
airmass = opts.airmass
aerosol = opts.aerosol
pwv = opts.pwv
ozone = opts.ozone
beta = opts.beta
plot_results = opts.plot_results
outName = opts.outName
outDir = opts.outDir

if outName != 'None':
    checkDir(outDir)

#
parNames = ['airmass', 'pwv', 'aerosol', 'ozone', 'beta']
parValues = [airmass, pwv, aerosol, ozone, beta]
params = dict(zip(parNames, parValues))

# fit
fitparNames = ['airmass', 'pwv', 'ozone', 'aerosol', 'beta']
fitparValues = [1.2, 4., 100, 0.1, 1.4]
fitparLimits = [(1., 3.), (0.5, 12.), (100., 500.), (0., 1.), (0., 5.)]

fitparNames = ['pwv', 'ozone', 'aerosol']
fitparValues = [4., 270, 0.1]
fitparLimits = [(0.5, 12.), (100., 500.), (0., 1.)]

fitparNames = ['pwv', 'ozone', 'aerosol']
fitparValues = [4.0, 270, 0.1]
fitparLimits = [(1., 12.), (100., 500.), (0., 1.)]

myfit = Fit_Atmos(params,
                  fitparNames=fitparNames,
                  fitparValues=fitparValues,
                  fitparLimits=fitparLimits)

resfit = myfit()

if outName != 'None':
    pp = {}
    for key, vals in resfit.items():
        pp[key] = [vals]
    for key, vals in params.items():
        pp[key] = [vals]
    df = pd.DataFrame.from_dict(pp)
    fullOut = '{}/{}'.format(outDir, outName)
    df.to_hdf(fullOut, key='fit_atmos')

# plot results
if plot_results:
    plot_atmos_trans(atmosDir, resfit, params, fitparNames)
