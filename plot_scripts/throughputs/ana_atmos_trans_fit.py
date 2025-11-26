#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:19:47 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import glob
from optparse import OptionParser

parser = OptionParser(description='Script to analyze atmos trans fits')

parser.add_option('--theDir', type=str, default='../fit_lsst_atmos',
                  help='file dir [%default]')

opts, args = parser.parse_args()
theDir = opts.theDir

df = pd.DataFrame()

fis = glob.glob('{}/*.hdf5'.format(theDir))

for fi in fis:
    dfa = pd.read_hdf(fi)
    df = pd.concat((df, dfa))

what = ['mean', 'std']

rr = df.agg({'pwv_fit': what,
             'ozone_fit': what,
             'aerosol_fit': what})

print(rr)
