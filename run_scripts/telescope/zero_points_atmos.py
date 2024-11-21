#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:49:10 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_telmodel.sn_throughputs_new import Throughputs
import numpy as np
import pandas as pd


def get_random_values(ntrials, mean_values, sigma_values):

    rnd = {}
    for key in mean_values.keys():
        rnd[key] = []
    for i in range(ntrials):
        for key, vals in mean_values.items():
            sigma = sigma_values[key]
            vv = vals+np.random.normal(0., sigma)
            rnd[key].append(vv)

    res = pd.DataFrame.from_dict(rnd)

    return res


def get_values(names, values):

    return dict(zip(names, values))


def zp(data, params):

    throughput = params['throughput']
    zp_dict = {}
    bands = list('ugrizy')
    zp_dict = dict(zip(bands, [[], [], [], [], [], []]))

    print('zp dict', zp_dict, bands, len(bands))

    for i, row in data.iterrows():
        throughput.reset_data()
        throughput.new_atmosphere(airmass=row['airmass'],
                                  aerosol=row['aerosol'],
                                  pwv=row['pwv'],
                                  oz=row['ozone'],
                                  beta=row['beta'])

        # tel.mean_wave()
        for b in 'ugrizy':
            # mean_wave = tel.mean_wavelength[b]
            zpb = throughput.zp(b)
            zp_dict[b].append(zpb)

    res = pd.DataFrame.from_dict(zp_dict)

    return res


tel_dir = 'throughputs'
through_dir = 'baseline'
tag = '1.9'

tel_dir = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(tel_dir, through_dir)

site_name = 'LSST'
pressure = 743.

throughput = Throughputs(
    tel_dir=through_dir, site_name=site_name, pressure=pressure)

par_names = ['airmass', 'pwv', 'ozone', 'beta', 'aerosol']

mean_values = get_values(par_names, [1.2, 4.0, 300., 0.05, 0.05])
sigma_values = get_values(par_names, [0.01, 0.2, 10., 0.0, 0.001])

ntrials = 1000

param_values = get_random_values(ntrials, mean_values, sigma_values)

print(param_values)

params = {}

params['throughput'] = throughput

zp_values = zp(param_values[:10], params)

print(zp_values)
