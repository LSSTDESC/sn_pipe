#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:55:33 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_io import add_parser
from sn_plotter_analysis import plt

parser = OptionParser(description='lsst matched auxtel analysis')

confDict = make_dict_from_config('input_script', 'lsst_auxtel_analysis.txt')

# parser for simulation parameters : 'dynamical' generation
add_parser(parser, confDict)

opts, args = parser.parse_args()

params=vars(opts)

#load data

fName = '{}/lsst_plus_auxtel_{}_{}_{}.hdf5'.format(params['dataDir'],
                                                   params['dbName'],
                                                   params['auxtel_type'],
                                                   params['year'])

df = pd.read_hdf(fName)

print(df)

fig, ax = plt.subplots()

ax.hist(df['delta_mjd']*24*60.,histtype='step',bins=100)
#ax.hist(df['delta_r'],histtype='step',bins=50)
plt.show()