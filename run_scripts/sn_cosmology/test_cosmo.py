import h5py
from astropy.table import Table, vstack
from sn_cosmology.cosmo_fit import CosmoFit, fom
import numpy as np


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
            Results of the array x in the funcion chosen in the parameters via the 
            parameters given.
        '''

        # instance of the cosmology model
        to_import = 'from astropy.cosmology import {}'.format(self.cosmo_model)
        exec(to_import)

        import copy
        # set default parameters
        parDict = copy.deepcopy(self.cosmo_default)

        # change the parameters that have to be changes

        for i, val in enumerate(self.parNames):
            parDict[val] = parameters[i]

        cosmo = eval(
            '{}(H0=70, Om0={}, Ode0=0.7,w0={}, wa={})'.format(self.cosmo_model,
                                                              parDict['Om0'],
                                                              parDict['w0'],
                                                              parDict['wa']))
        f = cosmo.distmod(self.x).value
        self.h = np.max(f) * (10**-7)

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
        f = self.y - self.fit_function(*parameters)
        # Matrix calculation of Xisquare
        X_mat = np.matmul(f * self.sigma**-2, f)
        # prior to be set here

        if self.prior:
            for i, name in enumerate(self.parNames):
                if name in self.prior.keys():
                    ref_val = self.prior[name][0]
                    sigma_val = self.prior[name][1]
                    X_mat += ((parameters[i] - ref_val)**2)/sigma_val**2

        return X_mat


fName = 'hdf5_simu.hdf5'

fFile = h5py.File(fName, 'r')

keys = list(fFile.keys())
parNames = ['w0', 'wa', 'Om0']
parNamesb = ['w0', 'wa', 'Om']
prior = dict(zip(['Om0'], [[0.3, 0.0073]]))

# parNames = ['w0', 'Om0']
# parNamesb = ['w0', 'Om']
prior = {}

corresp = dict(zip(parNames, parNamesb))
for key in keys:
    data = Table.read(fFile, path=key)
    print('data', key)
    print(data)
    myfit = MyFit(data['z'], data['mu'], data['sigma'],
                  parNames=parNames, prior=prior, par_protect_fit=[])
    params = []
    for pp in parNames:
        params.append(data.meta[corresp[pp]])

    dict_fit = myfit.minuit_fit(params)
    fisher_cov = myfit.covariance_fisher(params)
    dict_fit.update(fisher_cov)
    print(dict_fit)

    for vv in ['fit', 'fisher']:
        cov_a = dict_fit['Cov_w0_w0_{}'.format(vv)]
        cov_b = dict_fit['Cov_wa_wa_{}'.format(vv)]
        cov_ab = dict_fit['Cov_wa_w0_{}'.format(vv)]
        fom_val = fom(cov_a, cov_b, cov_ab)
        print(vv, fom_val)

    break
