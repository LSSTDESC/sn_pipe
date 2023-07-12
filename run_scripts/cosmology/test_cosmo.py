from optparse import OptionParser
import h5py
from astropy.table import Table
from sn_cosmology.cosmo_fit import CosmoFit, fom
import numpy as np
import pandas as pd
import time
from sn_tools.sn_utils import multiproc


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
            for i, val in enumerate(self.fitparNames):
                parDict[val] = parameters[i]

        cosmo = eval(
            '{}(H0=70, Om0={}, Ode0=0.7,w0={}, wa={})'.format(self.cosmo_model,
                                                              parDict['Om0'],
                                                              parDict['w0'],
                                                              parDict['wa']))
        f = cosmo.distmod(self.z).value
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
        f = self.mu - self.fit_function(*parameters)
        # Matrix calculation of Xisquare
        X_mat = np.matmul(f * self.sigma_mu**-2, f)
        # prior to be set here

        if not self.prior.empty:
            idx = self.prior['varname'].isin(self.fitparNames)
            for io, row in self.prior[idx].iterrows():
                ref_val = row['refvalue']
                sigma_val = row['sigma']
                i = self.fitparNames.index(row['varname'])
                if len(parameters) > 0:
                    X_mat += ((parameters[i] - ref_val)**2)/sigma_val**2

            """
            for i, name in enumerate(self.fitparNames):
                idx = self.prior['varname'] == name
                sel = self.prior[idx]
                if len(sel) > 0:
                    ref_val = sel['refvalue']
                    sigma_val = sel['sigma']
                    X_mat += ((parameters[i] - ref_val)**2)/sigma_val**2
            """
        return X_mat


def process_data_indiv(key, data, fitparNames, prior, par_protect_fit):
    """


    Parameters
    ----------
    key : str
        dataID.
    data : array
        data to process.
    fitparNames : list(str)
        list of parameters to fit
    prior : dict
        priors for the chisquare.
    par_protect_fit : list(str)
        list of fitparameters to protect

    Returns
    -------
    pandas df
        array with the results.

    """

    dict_out = {}
    dict_out['runNum'] = key
    dict_out['config'] = key.split('_')[0]

    dict_out.update(data.meta)
    dataValues = (data['z'], data['mu'], data['sigma'])
    dataNames = ['z', 'mu', 'sigma_mu']
    myfit = MyFit(dataValues, dataNames,
                  fitparNames=fitparNames, prior=prior,
                  par_protect_fit=par_protect_fit)
    params = []
    for pp in fitparNames:
        params.append(data.meta[pp])

    time_ref = time.time()
    dict_fit = myfit.minuit_fit(params)

    # get the Chisquare
    fitpars = []
    for pp in fitparNames:
        fitpars.append(dict_fit['{}_fit'.format(pp)])
    dict_fit['Chi2_fit'] = myfit.xi_square(*fitpars)
    dict_fit['Chi2_fisher'] = myfit.xi_square(*[])
    dict_fit['NDoF'] = len(data)-len(fitparNames)
    dict_fit['time_fit'] = time.time()-time_ref
    time_ref = time.time()
    fisher_cov = myfit.covariance_fisher(params)
    dict_fit['time_fisher'] = time.time()-time_ref
    dict_fit.update(fisher_cov)

    for vv in ['fit', 'fisher']:
        cov_a = dict_fit['Cov_w0_w0_{}'.format(vv)]
        cov_b = dict_fit['Cov_wa_wa_{}'.format(vv)]
        cov_ab = dict_fit['Cov_wa_w0_{}'.format(vv)]
        fom_val = fom(cov_a, cov_b, cov_ab)
        dict_fit['FoM_{}'.format(vv)] = fom_val

    dict_out.update(dict_fit)

    dict_res = {}
    for key, vals in dict_out.items():
        dict_res[key] = [vals]

    return pd.DataFrame.from_dict(dict_res)


def process(keys, params, j, output_q=None):
    """
    Function to process multiple data

    Parameters
    ----------
    keys : list(str)
        data keys to process.
    params : dict
        parameters for the processing.
    j : int
        multiprocessing num
    output_q : multiprocessing queue, optional
        where to put the data if not None. The default is None.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """

    fFile = params['fFile']
    prior = params['prior']
    par_protect_fit = params['par_protect_fit']
    fitparNames = params['fitparNames']
    res = pd.DataFrame()
    for i, key in enumerate(keys):
        data = Table.read(fFile, path=key)
        data.meta['Om0'] = data.meta['Om']
        res_proc = process_data_indiv(
            key, data, fitparNames, prior, par_protect_fit)
        res = pd.concat((res_proc, res))

    if output_q is not None:
        output_q.put({j: res})
    else:
        return res


parser = OptionParser()

parser.add_option("--fName", type="str",
                  default='hdf5_simu.hdf5', help="file to process [%default]")
parser.add_option("--priorFile", type="str", default='None',
                  help="to set prior or not[%default]")
parser.add_option("--fitparams", type="str",
                  default='w0,wa,Om0', help="parameters to fit [%default]")
parser.add_option("--nproc", type="int", default=4,
                  help="nproc for multiprocessing [%default]")

opts, args = parser.parse_args()

fName = opts.fName
priorFile = opts.priorFile
fitpars = opts.fitparams
nproc = opts.nproc

fFile = h5py.File(fName, 'r')

keys = list(fFile.keys())[:2]
print('number of configs', len(keys))
fitparNames = fitpars.split(',')

theprior = pd.DataFrame()

if priorFile != 'None':
    theprior = pd.read_csv(priorFile)

par_protect_fit = ['Om0']

ff = '_'.join(fitparNames)
outName = 'cosmo_{}'.format(ff)
if priorFile != 'None':
    outName += '_{}'.format(priorFile.split('.csv')[0])
else:
    outName += '_noprior'

outName += '.hdf5'

params = {}
params['fFile'] = fFile
params['prior'] = theprior
params['par_protect_fit'] = par_protect_fit
params['fitparNames'] = fitparNames

time_ref = time.time()
res = multiproc(keys, params, process, nproc)
tt = time.time()-time_ref
print(res, tt)
res['priorFile'] = priorFile
res.to_hdf(outName, key='cosmofit')
