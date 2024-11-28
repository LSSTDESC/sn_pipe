#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:49:10 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import matplotlib.pyplot as plt
from sn_telmodel.sn_throughputs_new import Throughputs
import numpy as np
import pandas as pd
from sn_tools.sn_utils import multiproc
import time


class Sigma_zp:
    def __init__(self, tel_dir, site_name='LSST', pressure=743.,
                 par_names=['airmass', 'pwv', 'ozone', 'beta', 'aerosol'],
                 par_means=[1.2, 4.0, 300., 0.05, 0.05],
                 par_sigmas=[0.01, 0.2, 10., 0.0, 0.001]):
        """
        class to estimate sigma_zp according to atmospheric parameters variation

        Parameters
        ----------
        tel_dir : str
            Telescope dir.
        site_name : str, optional
            Site name for observations. The default is 'LSST'.
        pressure : float, optional
            Site pressure. The default is 743..
        par_names : list(str), optional
            parameter list. 
            The default is ['airmass', 'pwv', 'ozone', 'beta', 'aerosol'].
        par_means : list(float), optional
            parameter means. The default is [1.2, 4.0, 300., 0.05, 0.05].
        par_sigmas : list(float), optional
            parameters sigmas. The default is [0.01, 0.2, 10., 0.0, 0.001].

        Returns
        -------
        None.

        """

        self.throughput = Throughputs(
            tel_dir=through_dir,
            site_name=site_name, pressure=pressure)

        self.mean_values = self.get_values(par_names, par_means)
        self.sigma_values = self.get_values(par_names, par_sigmas)

    def get_values(self, names, values):
        """
        Method to transform two lists to a dict

        Parameters
        ----------
        names : list(str)
            List of names.
        values : list(float)
            List of values.

        Returns
        -------
        dict
            Resulting dict.

        """

        return dict(zip(names, values))

    def __call__(self, ntrials=1000, nproc=8):
        """
        Main method for data processing

        Parameters
        ----------
        ntrials : int, optional
            number of random parameter choices. The default is 1000.
        nproc : int, optional
            number of procs to use for processing. The default is 8.

        Returns
        -------
        df : pandas df
            Result.

        """

        # get random values
        param_values = self.get_random_values(ntrials)

        params = {}

        params['throughput'] = self.throughput
        zp_values = multiproc(param_values, params, self.zp, nproc)

        vv = zp_values.mean().to_list()
        cols = zp_values.columns.to_list()
        colsb = list(map(lambda x: 'mean_zp_' + x, cols))
        df = pd.DataFrame([vv], columns=colsb)
        colsc = list(map(lambda x: 'std_zp_' + x, cols))
        df[colsc] = zp_values.std().to_list()

        # add atmospheric parameters
        df = self.concat(df, self.mean_values, 'mean')
        df = self.concat(df, self.sigma_values, 'sigma')

        return df

    def concat(self, dfa, thedict, prefix):
        """
        Method to concat df with dict transformed as df

        Parameters
        ----------
        dfa : pandas df
            original df.
        thedict : dict
            Data to merge.
        prefix : str
            prefix for column names.

        Returns
        -------
        dfa : pandas df
            Resulting merged df.

        """

        dfb = self.make_df(thedict, prefix)
        dfa = pd.concat((dfa, dfb), axis=1)

        return dfa

    def make_df(self, thedict, prefix='mean'):
        """
        Method to create a df from dict with column name change

        Parameters
        ----------
        thedict : dict
            Data to process.
        prefix : str, optional
            prefix to add to col names. The default is 'mean'.

        Returns
        -------
        dfa : pandas df
            Result.

        """

        cols = thedict.keys()
        colsb = list(map(lambda x: '{}_'.format(prefix) + x, cols))
        dfa = pd.DataFrame([thedict.values()], columns=colsb)

        return dfa

    def get_random_values(self, ntrials):
        """
        Method to estimate random values for atmospheric parameters

        Parameters
        ----------
        ntrials : int
            number of random parameter choices.

        Returns
        -------
        res : pandas df
            random values for atmospheric parameters.

        """

        rnd = {}
        for key in self.mean_values.keys():
            rnd[key] = []
        for i in range(ntrials):
            for key, vals in self.mean_values.items():
                sigma = self.sigma_values[key]
                vv = vals+np.random.normal(0., sigma)
                rnd[key].append(vv)

        res = pd.DataFrame.from_dict(rnd)

        return res

    def zp(self, data, params, j=0, output_q=None):
        """
        Method to estimate zero points for atmospheric parameters (data)

        Parameters
        ----------
        data : pandas df
            Atmospheric parameters.
        params : dict
            Method parameters.
        j : int, optional
            int for multiprocessing. The default is 0.
        output_q : multiprocessing queue, optional
            Where to store the results. The default is None.

        Returns
        -------
        pandas df
            zp results for each band.

        """

        throughput = params['throughput']
        zp_dict = {}
        bands = list('ugrizy')
        zp_dict = dict(zip(bands, [[], [], [], [], [], []]))
        mean_wave_dict = dict(zip(bands, [[], [], [], [], [], []]))

        for i, row in data.iterrows():
            throughput.reset_data()
            throughput.new_atmosphere(airmass=row['airmass'],
                                      aerosol=row['aerosol'],
                                      pwv=row['pwv'],
                                      oz=row['ozone'],
                                      beta=row['beta'])

            throughput.mean_wave()
            for b in 'ugrizy':
                # mean_wave = tel.mean_wavelength[b]
                zpb = throughput.zp(b)
                zp_dict[b].append(zpb)
                mean_wave_dict[b].append(throughput.mean_wavelength[b])

        res = pd.DataFrame.from_dict(zp_dict)
        print('rrr', mean_wave_dict)

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res


def get_combi(thedict, parList):
    """
    Function to build a df of combination of parameters

    Parameters
    ----------
    thedict : dict
        input dict with parameters.
    parList : str
        parameter list.

    Returns
    -------
    df : pandas df
        output data: all the possible parameter combinations.

    """

    df = pd.DataFrame()
    for vv in parList:
        dfa = pd.DataFrame(thedict[vv], columns=[vv])
        if len(df) > 0:
            df = df.merge(dfa, how='cross')
        else:
            df = dfa

    return df


time_ref = time.time()
tel_dir = 'throughputs'
through_dir = 'baseline'
tag = '1.9'

tel_dir = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(tel_dir, through_dir)

site_name = 'LSST'
pressure = 743.
bands = 'ugrizy'

par_names = ['airmass', 'pwv', 'ozone', 'beta', 'aerosol']
par_means = [1.2, 4.0, 300., 0.05, 0.05]
par_sigmas = [0.01, 0.2, 10., 0.0, 0.001]

dict_sigma = {}
dict_sigma['airmass'] = list(np.arange(0.01, 0.11, 0.01))
dict_sigma['pwv'] = list(np.arange(0.01, 0.5, 0.01))
dict_sigma['ozone'] = list(np.arange(8., 21, 1.))
dict_sigma['aerosol'] = list(np.arange(0.01, 0.06, 0.01))
dict_sigma['beta'] = [0.]

combi_sigma = get_combi(dict_sigma, par_names)

print(combi_sigma)

df = pd.DataFrame()

for i, row in combi_sigma[:100].iterrows():
    par_sigmas = row[par_names].to_list()
    sigma_zp = Sigma_zp(tel_dir, site_name, pressure,
                        par_names, par_means, par_sigmas)

    res = sigma_zp(ntrials=100)

    df = pd.concat((df, res))


print('finally', df)
print('end of processing', time.time()-time_ref)

for b in bands:
    fig, ax = plt.subplots()
    ax.hist(df['std_zp_{}'.format(b)], histtype='step', bins=20)

plt.show()
